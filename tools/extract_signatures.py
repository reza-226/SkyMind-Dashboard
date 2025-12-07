#!/usr/bin/env python3
"""
Script to extract and save all function/class signatures
برای جلوگیری از خطاهای تکراری
"""

import ast
import inspect
import importlib.util
import sys
from pathlib import Path
import json
from datetime import datetime

class SignatureExtractor:
    """Extract signatures from Python files"""
    
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.signatures = {}
    
    def extract_from_source(self, filepath):
        """Extract signatures using AST parsing"""
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return {'error': str(e)}
        
        signatures = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                signatures[class_name] = {'type': 'class', 'methods': {}}
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_name = item.name
                        args = [arg.arg for arg in item.args.args]
                        defaults = []
                        
                        # Extract defaults
                        if item.args.defaults:
                            num_defaults = len(item.args.defaults)
                            for i, default in enumerate(item.args.defaults):
                                arg_idx = len(args) - num_defaults + i
                                if isinstance(default, ast.Constant):
                                    defaults.append((args[arg_idx], default.value))
                                elif isinstance(default, ast.Num):
                                    defaults.append((args[arg_idx], default.n))
                                else:
                                    defaults.append((args[arg_idx], repr(default)))
                        
                        signatures[class_name]['methods'][method_name] = {
                            'args': args,
                            'defaults': defaults
                        }
            
            elif isinstance(node, ast.FunctionDef):
                func_name = node.name
                if func_name not in signatures:
                    args = [arg.arg for arg in node.args.args]
                    signatures[func_name] = {
                        'type': 'function',
                        'args': args
                    }
        
        return signatures
    
    def extract_from_runtime(self, module_path, class_name=None):
        """Extract actual runtime signature"""
        try:
            spec = importlib.util.spec_from_file_location("module", module_path)
            module = importlib.util.module_from_spec(spec)
            sys.path.insert(0, str(module_path.parent))
            spec.loader.exec_module(module)
            
            if class_name:
                cls = getattr(module, class_name)
                sig = inspect.signature(cls.__init__)
                return {
                    'class': class_name,
                    'method': '__init__',
                    'signature': str(sig),
                    'parameters': list(sig.parameters.keys())
                }
            else:
                return {'error': 'No class specified'}
                
        except Exception as e:
            return {'error': str(e)}
    
    def scan_directory(self, directory, pattern='*.py'):
        """Scan directory for Python files"""
        directory = Path(directory)
        results = {}
        
        for filepath in directory.rglob(pattern):
            relative_path = filepath.relative_to(self.project_root)
            print("Scanning: " + str(relative_path))
            
            # AST-based extraction
            ast_sigs = self.extract_from_source(filepath)
            
            results[str(relative_path)] = {
                'path': str(filepath),
                'ast_signatures': ast_sigs
            }
            
            # Try runtime extraction for important classes
            if 'actor_network' in str(filepath).lower():
                runtime_sig = self.extract_from_runtime(filepath, 'ActorNetwork')
                results[str(relative_path)]['runtime_ActorNetwork'] = runtime_sig
            
            if 'critic_network' in str(filepath).lower():
                runtime_sig = self.extract_from_runtime(filepath, 'CriticNetwork')
                results[str(relative_path)]['runtime_CriticNetwork'] = runtime_sig
        
        return results
    
    def save_results(self, output_file='project_signatures.json'):
        """Save signatures to JSON file"""
        output_path = self.project_root / output_file
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'signatures': self.signatures
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print("\nSignatures saved to: " + str(output_path))
        return output_path
    
    def generate_markdown_report(self):
        """Generate human-readable markdown report"""
        output_path = self.project_root / 'SIGNATURES_REFERENCE.md'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Project Signatures Reference\n\n")
            timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            line = "**Generated:** " + timestamp_str + "\n\n"
            f.write(line)
            f.write("---\n\n")
            
            for filepath, data in self.signatures.items():
                header = "## " + filepath + "\n\n"
                f.write(header)
                
                # AST signatures
                if 'ast_signatures' in data:
                    for name, sig_data in data['ast_signatures'].items():
                        if sig_data.get('type') == 'class':
                            class_header = "### Class: `" + name + "`\n\n"
                            f.write(class_header)
                            
                            for method, method_data in sig_data.get('methods', {}).items():
                                args = method_data['args']
                                defaults = method_data.get('defaults', [])
                                
                                args_str = ', '.join(args)
                                method_sig = "#### `" + method + "(" + args_str + ")`\n\n"
                                f.write(method_sig)
                                
                                if defaults:
                                    f.write("**Defaults:**\n")
                                    for arg, val in defaults:
                                        default_line = "- `" + str(arg) + "=" + str(val) + "`\n"
                                        f.write(default_line)
                                f.write("\n")
                
                # Runtime signatures
                if 'runtime_ActorNetwork' in data:
                    rt = data['runtime_ActorNetwork']
                    f.write("### Runtime ActorNetwork.__init__\n\n")
                    sig_str = rt.get('signature', 'N/A')
                    f.write("
```python\n" + sig_str + "\n
```\n\n")
                    params_str = ', '.join(rt.get('parameters', []))
                    params_line = "**Parameters:** `" + params_str + "`\n\n"
                    f.write(params_line)
                
                if 'runtime_CriticNetwork' in data:
                    rt = data['runtime_CriticNetwork']
                    f.write("### Runtime CriticNetwork.__init__\n\n")
                    sig_str = rt.get('signature', 'N/A')
                    f.write("
```python\n" + sig_str + "\n
```\n\n")
                    params_str = ', '.join(rt.get('parameters', []))
                    params_line = "**Parameters:** `" + params_str + "`\n\n"
                    f.write(params_line)
                
                f.write("---\n\n")
        
        print("Markdown report saved to: " + str(output_path))
        return output_path


def main():
    """Main execution"""
    print("Signature Extraction Tool\n")
    print("="*60)
    
    # Set project root
    project_root = Path(__file__).parent.parent
    print("Project root: " + str(project_root) + "\n")
    
    # Initialize extractor
    extractor = SignatureExtractor(project_root)
    
    # Scan directories
    directories = [
        'models/actor_critic',
        'models',
        'environment',
        'evaluation'
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        if dir_path.exists():
            print("\nScanning: " + directory)
            results = extractor.scan_directory(dir_path)
            extractor.signatures.update(results)
    
    # Save results
    print("\n" + "="*60)
    json_path = extractor.save_results()
    md_path = extractor.generate_markdown_report()
    
    print("\nDone!")
    print("\nFiles created:")
    print("   1. " + str(json_path) + " (machine-readable)")
    print("   2. " + str(md_path) + " (human-readable)")
    print("\nUse these files as reference to avoid signature mismatches!")


if __name__ == '__main__':
    main()
