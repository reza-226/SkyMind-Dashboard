import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
from arabic_reshaper import reshape
from bidi.algorithm import get_display

# ========================================
# تنظیمات فونت فارسی
# ========================================

plt.rcParams['font.family'] = 'B Nazanin'  # یا 'Tahoma' یا 'Vazir'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False  # حل مشکل علامت منفی

def fix_persian(text):
    """تبدیل متن فارسی برای نمایش صحیح"""
    return get_display(reshape(text))

# ========================================
# رنگ‌ها
# ========================================

COLORS = {
    'uav': '#3498db',
    'critic': '#e74c3c',
    'action': '#2ecc71',
    'arrow': '#95a5a6',
    'background': '#ecf0f1',
    'text': '#2c3e50'
}

# ========================================
# تابع 1: Training Phase
# ========================================

def create_training_phase():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # عنوان اصلی
    ax.text(5, 11.5, fix_persian('مرحله آموزش - Centralized Training'), 
            ha='center', va='top', fontsize=18, fontweight='bold', color=COLORS['text'])
    ax.text(5, 11, fix_persian('(CTDE: Centralized Training with Decentralized Execution)'),
            ha='center', va='top', fontsize=12, color='gray')
    
    # UAVs (5 عدد)
    uav_positions = [(1, 7), (2.5, 7), (4, 7), (5.5, 7), (7, 7)]
    for i, (x, y) in enumerate(uav_positions):
        # UAV Box
        uav_box = FancyBboxPatch((x-0.4, y-0.5), 0.8, 1, 
                                  boxstyle="round,pad=0.1", 
                                  edgecolor=COLORS['uav'], 
                                  facecolor='lightblue', 
                                  linewidth=2)
        ax.add_patch(uav_box)
        
        # متن UAV
        ax.text(x, y+0.2, f'UAV {i+1}', ha='center', va='center', 
                fontsize=10, fontweight='bold', color=COLORS['text'])
        ax.text(x, y-0.1, fix_persian(f'عامل {i+1}'), ha='center', va='center', 
                fontsize=9, color='gray')
        
        # Actor Network
        actor_box = FancyBboxPatch((x-0.35, y-1.8), 0.7, 0.8,
                                    boxstyle="round,pad=0.05",
                                    edgecolor=COLORS['action'],
                                    facecolor='lightgreen',
                                    linewidth=1.5)
        ax.add_patch(actor_box)
        ax.text(x, y-1.5, fix_persian('شبکه Actor'), ha='center', va='center',
                fontsize=8, fontweight='bold')
        ax.text(x, y-1.7, f'(θᵢ)', ha='center', va='center', fontsize=7, color='gray')
        
        # فلش از UAV به Actor
        arrow1 = FancyArrowPatch((x, y-0.5), (x, y-1.0),
                                  arrowstyle='->', mutation_scale=15,
                                  color=COLORS['arrow'], linewidth=1.5)
        ax.add_patch(arrow1)
        ax.text(x+0.3, y-0.75, fix_persian('مشاهده'), fontsize=7, color='gray')
        
        # فلش از Actor به پایین (Action)
        arrow2 = FancyArrowPatch((x, y-1.8), (x, y-2.3),
                                  arrowstyle='->', mutation_scale=15,
                                  color=COLORS['action'], linewidth=2)
        ax.add_patch(arrow2)
        ax.text(x+0.3, y-2.1, fix_persian('عمل'), fontsize=7, color=COLORS['action'], fontweight='bold')

    # Centralized Critic
    critic_x, critic_y = 5, 3.5
    critic_box = FancyBboxPatch((critic_x-1.2, critic_y-0.6), 2.4, 1.2,
                                 boxstyle="round,pad=0.1",
                                 edgecolor=COLORS['critic'],
                                 facecolor='#ffcccc',
                                 linewidth=3)
    ax.add_patch(critic_box)
    
    ax.text(critic_x, critic_y+0.3, fix_persian('شبکه Critic متمرکز'), 
            ha='center', va='center', fontsize=12, fontweight='bold', color=COLORS['critic'])
    ax.text(critic_x, critic_y, fix_persian('(دریافت تمام حالات و اعمال)'),
            ha='center', va='center', fontsize=9, color='gray')
    ax.text(critic_x, critic_y-0.3, 'Q(s₁,...,s₅, a₁,...,a₅)', 
            ha='center', va='center', fontsize=8, style='italic')
    
    # فلش‌های ورودی به Critic
    for i, (x, _) in enumerate(uav_positions):
        arrow = FancyArrowPatch((x, 4.5), (critic_x, critic_y+0.6),
                                 arrowstyle='->', mutation_scale=12,
                                 color=COLORS['arrow'], linewidth=1.5,
                                 linestyle='dashed')
        ax.add_patch(arrow)
    
    # فلش‌های بازخورد (Gradient)
    for i, (x, y) in enumerate(uav_positions):
        arrow_back = FancyArrowPatch((critic_x, critic_y-0.6), (x, y-1.0),
                                      arrowstyle='->', mutation_scale=12,
                                      color='red', linewidth=2,
                                      linestyle='dotted')
        ax.add_patch(arrow_back)
    
    ax.text(critic_x-1.5, 2.5, fix_persian('گرادیان بهینه‌سازی'), 
            fontsize=9, color='red', fontweight='bold')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='lightblue', edgecolor=COLORS['uav'], label=fix_persian('عامل UAV')),
        mpatches.Patch(facecolor='lightgreen', edgecolor=COLORS['action'], label=fix_persian('شبکه Actor محلی')),
        mpatches.Patch(facecolor='#ffcccc', edgecolor=COLORS['critic'], label=fix_persian('Critic متمرکز')),
        mpatches.Patch(facecolor='none', edgecolor='red', linestyle='dotted', label=fix_persian('بازخورد گرادیان'))
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9, frameon=True)
    
    # توضیحات
    ax.text(5, 1.5, fix_persian('✓ آموزش: Critic متمرکز تمام اطلاعات را می‌بیند'), 
            ha='center', fontsize=10, color='green', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.text(5, 0.8, fix_persian('✓ هر Actor فقط مشاهدات محلی خود را دارد'), 
            ha='center', fontsize=10, color='blue', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    return fig

# ========================================
# تابع 2: Execution Phase
# ========================================

def create_execution_phase():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # عنوان
    ax.text(5, 11.5, fix_persian('مرحله اجرا - Decentralized Execution'), 
            ha='center', va='top', fontsize=18, fontweight='bold', color=COLORS['text'])
    ax.text(5, 11, fix_persian('(Critic حذف شده - هر UAV مستقل عمل می‌کند)'),
            ha='center', va='top', fontsize=12, color='gray')
    
    # UAVs
    uav_positions = [(1, 7), (2.5, 7), (4, 7), (5.5, 7), (7, 7)]
    for i, (x, y) in enumerate(uav_positions):
        # UAV Box
        uav_box = FancyBboxPatch((x-0.4, y-0.5), 0.8, 1,
                                  boxstyle="round,pad=0.1",
                                  edgecolor=COLORS['uav'],
                                  facecolor='lightblue',
                                  linewidth=2)
        ax.add_patch(uav_box)
        
        ax.text(x, y+0.2, f'UAV {i+1}', ha='center', va='center',
                fontsize=10, fontweight='bold', color=COLORS['text'])
        ax.text(x, y-0.1, fix_persian(f'عامل {i+1}'), ha='center', va='center',
                fontsize=9, color='gray')
        
        # Actor (Trained)
        actor_box = FancyBboxPatch((x-0.35, y-1.8), 0.7, 0.8,
                                    boxstyle="round,pad=0.05",
                                    edgecolor=COLORS['action'],
                                    facecolor='lightgreen',
                                    linewidth=1.5)
        ax.add_patch(actor_box)
        ax.text(x, y-1.4, fix_persian('Actor آموخته‌شده'), ha='center', va='center',
                fontsize=8, fontweight='bold')
        ax.text(x, y-1.65, f'(θᵢ*)', ha='center', va='center', fontsize=7, color='green')
        
        # فلش‌ها
        arrow1 = FancyArrowPatch((x, y-0.5), (x, y-1.0),
                                  arrowstyle='->', mutation_scale=15,
                                  color=COLORS['arrow'], linewidth=1.5)
        ax.add_patch(arrow1)
        
        arrow2 = FancyArrowPatch((x, y-1.8), (x, y-2.3),
                                  arrowstyle='->', mutation_scale=15,
                                  color=COLORS['action'], linewidth=2)
        ax.add_patch(arrow2)
        
        ax.text(x+0.3, y-0.75, fix_persian('مشاهده'), fontsize=7, color='gray')
        ax.text(x+0.3, y-2.1, fix_persian('تصمیم'), fontsize=7, color=COLORS['action'])
    
    # Critic (حذف شده)
    critic_x, critic_y = 5, 4
    critic_box = FancyBboxPatch((critic_x-1.2, critic_y-0.6), 2.4, 1.2,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='gray',
                                 facecolor='#f0f0f0',
                                 linewidth=2,
                                 linestyle='dashed')
    ax.add_patch(critic_box)
    
    ax.text(critic_x, critic_y+0.2, fix_persian('Critic (حذف شده)'), 
            ha='center', va='center', fontsize=12, fontweight='bold', color='gray')
    ax.text(critic_x, critic_y-0.2, fix_persian('✗ در اجرا نیازی نیست'), 
            ha='center', va='center', fontsize=9, color='red')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='lightblue', edgecolor=COLORS['uav'], label=fix_persian('عامل UAV مستقل')),
        mpatches.Patch(facecolor='lightgreen', edgecolor=COLORS['action'], label=fix_persian('Actor آموخته‌شده')),
        mpatches.Patch(facecolor='#f0f0f0', edgecolor='gray', linestyle='dashed', label=fix_persian('Critic غیرفعال'))
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9, frameon=True)
    
    # توضیحات
    ax.text(5, 2.5, fix_persian('✓ هر UAV با Actor محلی خود تصمیم می‌گیرد'), 
            ha='center', fontsize=10, color='green', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.text(5, 1.8, fix_persian('✓ نیازی به ارتباط متمرکز یا هماهنگی نیست'), 
            ha='center', fontsize=10, color='blue', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    return fig

# ========================================
# تابع 3: Comparison
# ========================================

def create_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Training Phase (سمت چپ)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title(fix_persian('مرحله آموزش (Training)'), fontsize=14, fontweight='bold', color='red')
    
    # Execution Phase (سمت راست)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title(fix_persian('مرحله اجرا (Execution)'), fontsize=14, fontweight='bold', color='green')
    
    # رسم ساده‌شده
    for ax, phase in [(ax1, 'train'), (ax2, 'exec')]:
        # UAVs
        for i, x in enumerate([2, 4, 6, 8]):
            circle = Circle((x, 8), 0.4, color='lightblue', ec='blue', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, 8, f'{i+1}', ha='center', va='center', fontweight='bold')
        
        if phase == 'train':
            # Critic مرکزی
            critic = FancyBboxPatch((3, 4), 4, 1.5, boxstyle="round,pad=0.1",
                                     edgecolor='red', facecolor='#ffcccc', linewidth=3)
            ax.add_patch(critic)
            ax.text(5, 4.75, fix_persian('Critic متمرکز'), ha='center', fontsize=10, fontweight='bold')
            
            # فلش‌ها
            for x in [2, 4, 6, 8]:
                ax.arrow(x, 7.6, 5-x, -3, head_width=0.2, head_length=0.2, 
                         fc='gray', ec='gray', linestyle='dashed')
                ax.arrow(5, 4, x-5, 3.6, head_width=0.2, head_length=0.2,
                         fc='red', ec='red', linestyle='dotted')
        else:
            # بدون Critic
            ax.text(5, 4.75, fix_persian('✗ Critic حذف شده'), ha='center', fontsize=12, 
                    color='red', bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.5))
    
    # جدول مقایسه
    table_data = [
        [fix_persian('ویژگی'), fix_persian('آموزش'), fix_persian('اجرا')],
        [fix_persian('Critic'), '✓ فعال', '✗ غیرفعال'],
        [fix_persian('ارتباط'), fix_persian('متمرکز'), fix_persian('غیرمتمرکز')],
        [fix_persian('پیچیدگی'), fix_persian('بالا'), fix_persian('پایین')],
        [fix_persian('سرعت'), fix_persian('کند'), fix_persian('سریع')]
    ]
    
    table = plt.table(cellText=table_data, cellLoc='center', loc='bottom',
                      bbox=[0.1, -0.4, 0.8, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#cccccc')
                cell.set_text_props(weight='bold')
    
    plt.tight_layout()
    return fig

# ========================================
# اجرا
# ========================================

if __name__ == '__main__':
    print('🚀 شروع تولید دیاگرام‌های MADDPG با پشتیبانی کامل فارسی...')
    print('='*70)
    
    # تولید دیاگرام‌ها
    print('\n📊 [1/3] در حال تولید: Training Phase...')
    fig1 = create_training_phase()
    fig1.savefig('MADDPG_Training_Phase_FA.png', dpi=300, bbox_inches='tight', facecolor='white')
    print('   ✅ ذخیره شد: MADDPG_Training_Phase_FA.png')
    
    print('\n📊 [2/3] در حال تولید: Execution Phase...')
    fig2 = create_execution_phase()
    fig2.savefig('MADDPG_Execution_Phase_FA.png', dpi=300, bbox_inches='tight', facecolor='white')
    print('   ✅ ذخیره شد: MADDPG_Execution_Phase_FA.png')
    
    print('\n📊 [3/3] در حال تولید: Comparison Diagram...')
    fig3 = create_comparison()
    fig3.savefig('MADDPG_Comparison_FA.png', dpi=300, bbox_inches='tight', facecolor='white')
    print('   ✅ ذخیره شد: MADDPG_Comparison_FA.png')
    
    print('\n' + '='*70)
    print('🎉 تمام دیاگرام‌ها با موفقیت تولید شدند!')
    print('='*70)
    print('\n📁 فایل‌های خروجی (با فونت فارسی):')
    print('   1️⃣  MADDPG_Training_Phase_FA.png')
    print('   2️⃣  MADDPG_Execution_Phase_FA.png')
    print('   3️⃣  MADDPG_Comparison_FA.png')
    print('\n💡 این نسخه با فونت فارسی کامل کار می‌کند!')
    print('='*70)
