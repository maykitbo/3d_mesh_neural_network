#!/bin/bash

# Check if MySQL is running
if ! systemctl is-active --quiet mysqld; then
    echo "Starting MySQL..."
    sudo systemctl start mysqld
else
    echo "MySQL is already running."
fi
