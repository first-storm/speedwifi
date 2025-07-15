#!/bin/bash

# runtest.sh - Batch run speedwifi.py test script
# Usage: runtest.sh -t <count> <output_path>

set -e  # Exit on error

# Default values
TEST_COUNT=1
OUTPUT_DIR=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPEEDWIFI_SCRIPT="$SCRIPT_DIR/speedwifi.py"

# Show help information
show_help() {
    echo "Usage: $0 -t <test_count> <output_path>"
    echo ""
    echo "Options:"
    echo "  -t <count>     Specify test count (default: 1)"
    echo "  -h, --help     Show this help message"
    echo ""
    echo "Arguments:"
    echo "  <output_path>  Directory path to save test results"
    echo ""
    echo "Examples:"
    echo "  $0 -t 3 /home/aoba/Downloads/blockhouse/k-g6-114/"
    echo "  $0 -t 5 ./test-results/"
    echo ""
    echo "Output file format: test-1.json, test-2.json, test-3.json ..."
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t)
            if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                TEST_COUNT="$2"
                shift 2
            else
                echo "Error: -t option requires a positive integer argument" >&2
                exit 1
            fi
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            echo "Error: Unknown option $1" >&2
            show_help
            exit 1
            ;;
        *)
            if [[ -z "$OUTPUT_DIR" ]]; then
                OUTPUT_DIR="$1"
                shift
            else
                echo "Error: Extra argument $1" >&2
                show_help
                exit 1
            fi
            ;;
    esac
done

# Check required arguments
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: Output path must be specified" >&2
    show_help
    exit 1
fi

if [[ $TEST_COUNT -lt 1 ]]; then
    echo "Error: Test count must be greater than 0" >&2
    exit 1
fi

# Check if speedwifi.py exists
if [[ ! -f "$SPEEDWIFI_SCRIPT" ]]; then
    echo "Error: speedwifi.py script not found: $SPEEDWIFI_SCRIPT" >&2
    exit 1
fi

# Check if Python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 command not found, please install Python 3" >&2
    exit 1
fi

# Create output directory if it does not exist
if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR" || {
        echo "Error: Failed to create directory $OUTPUT_DIR" >&2
        exit 1
    }
fi

# Ensure output directory path ends with /
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

echo "========================================"
echo "SpeedWiFi Batch Test Script"
echo "========================================"
echo "Test count: $TEST_COUNT"
echo "Output path: $OUTPUT_DIR"
echo "Test script: $SPEEDWIFI_SCRIPT"
echo "========================================"
echo ""

# Record start time
START_TIME=$(date +%s)
FAILED_TESTS=0

# Run tests
for ((i=1; i<=TEST_COUNT; i++)); do
    output_file="$OUTPUT_DIR/test-$i.json"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting test $i/$TEST_COUNT ..."
    echo "Output file: $output_file"
    
    # Run test
    if python3 "$SPEEDWIFI_SCRIPT" --json "$output_file" --bytes $((64*1024*1024)); then
        echo "✓ Test $i completed"
        
        # Check if file was created successfully
        if [[ -f "$output_file" ]]; then
            file_size=$(stat -c%s "$output_file" 2>/dev/null || echo "0")
            echo "  File size: ${file_size} bytes"
        else
            echo "⚠ Warning: Output file not created"
            ((FAILED_TESTS++))
        fi
    else
        echo "✗ Test $i failed"
        ((FAILED_TESTS++))
    fi
    
    echo ""
    
    # Wait before next test if not last
    if [[ $i -lt $TEST_COUNT ]]; then
        echo "Waiting 2 seconds before next test..."
        sleep 2
    fi
done

# Calculate total duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))

echo "========================================"
echo "Testing finished!"
echo "========================================"
echo "Total tests: $TEST_COUNT"
echo "Success: $((TEST_COUNT - FAILED_TESTS))"
echo "Failed: $FAILED_TESTS"
echo "Total time: ${DURATION_MIN}m${DURATION_SEC}s"
echo "Results saved in: $OUTPUT_DIR"
echo ""

# List generated files
echo "Generated test files:"
ls -la "$OUTPUT_DIR"/test-*.json 2>/dev/null || echo "  (No test files found)"

echo ""
if [[ $FAILED_TESTS -eq 0 ]]; then
    echo "🎉 All tests completed successfully!"
    exit 0
else
    echo "⚠ $FAILED_TESTS test(s) failed, please check error messages"
    exit 1
fi
