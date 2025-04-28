#!/bin/bash

# Install the gym_franka_sim package
pip install -e .

# Create a symbolic link to the lerobot package
# This is needed because the gym_franka_sim package imports from lerobot
LEROBOT_PATH="/home/khalil/Documents/Khalil/Projects/Embodied-AI/lerobot"
if [ -d "$LEROBOT_PATH" ]; then
    echo "Creating symbolic link to lerobot package"
    ln -sf "$LEROBOT_PATH" lerobot
else
    echo "Error: lerobot package not found at $LEROBOT_PATH"
    echo "Please update the LEROBOT_PATH variable in this script"
    exit 1
fi

echo "Installation complete!"
echo "You can now use the gym_franka_sim package"
echo "Try running one of the examples:"
echo "  python examples/simple_example.py"
echo "  python examples/push_example.py"
echo "  python examples/lerobot_integration.py"
