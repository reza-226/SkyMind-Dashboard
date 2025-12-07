#!/bin/bash

echo "=========================================="
echo "Quick Test for MADDPG Agents V2"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test imports
echo "Testing imports..."
python3 -c "from agents import ActorNetwork, CriticNetwork, ActionDecoder, MADDPGAgent; print('✓ All imports successful')" 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Import test passed${NC}"
else
    echo -e "${RED}✗ Import test failed${NC}"
    exit 1
fi

echo ""

# Run main tests
echo "Running test suite..."
python3 test_agents_v2.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi

echo ""
echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="
