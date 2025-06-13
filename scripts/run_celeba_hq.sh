#!/bin/bash

# Parallel execution script for CelebA-HQ dataset training
# This script runs multiple Python programs simultaneously on different GPUs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Configuration
n_iter=20
batch_size=32
dataset=celeba_hq
log_dir="logs/$dataset"
timestamp=$(date +"%Y%m%d_%H%M%S")

# Create log directory
mkdir -p "$log_dir"

# Array to store PIDs
pids=()
job_names=()
job_statuses=()  # Track job completion status: "running", "completed", "failed"
job_exit_codes=() # Track exit codes

# Function to run command in background with logging
run_parallel() {
    local target=$1
    local attack=$2
    local device=$3
    local job_name="${target}_${attack}_${device}"
    local log_file="$log_dir/${job_name}_${timestamp}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting job: $job_name on $device"
    
    # Run command in background with full logging
    {
        echo "=== Job: $job_name ==="
        echo "=== Target: $target, Attack: $attack, Device: $device ==="
        echo "=== Started at: $(date) ==="
        echo "=== Command: python run.py target=$target attack=$attack n_iter=$n_iter batch_size=$batch_size test_dataset=$dataset device=$device ==="
        echo ""
        
        python run.py \
            target=$target \
            attack=$attack \
            n_iter=$n_iter \
            batch_size=$batch_size \
            test_dataset=$dataset \
            device=$device
        
        local exit_code=$?
        echo ""
        echo "=== Finished at: $(date) ==="
        echo "=== Exit code: $exit_code ==="
        
        if [ $exit_code -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Job completed successfully: $job_name" >> "$log_dir/main_${timestamp}.log"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Job failed with exit code $exit_code: $job_name" >> "$log_dir/main_${timestamp}.log"
        fi
        
    } > "$log_file" 2>&1 &
    
    # Store PID and job name
    local pid=$!
    pids+=($pid)
    job_names+=($job_name)
    job_statuses+=("running")
    job_exit_codes+=(0)  # Initialize with 0, will be updated when job completes
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job $job_name started with PID: $pid" | tee -a "$log_dir/main_${timestamp}.log"
}

# Function to show running status
show_status() {
    echo ""
    echo "=== Job Status ==="
    for i in "${!pids[@]}"; do
        local pid=${pids[$i]}
        local job_name=${job_names[$i]}
        local job_status=${job_statuses[$i]}
        
        if [ "$job_status" == "running" ]; then
            if kill -0 "$pid" 2>/dev/null; then
                echo "✓ RUNNING: $job_name (PID: $pid)"
            else
                # Process has finished, check exit code from wait_all function
                local exit_code=${job_exit_codes[$i]}
                if [ $exit_code -eq 0 ]; then
                    echo "✓ COMPLETED: $job_name (PID: $pid)"
                    job_statuses[$i]="completed"
                else
                    echo "✗ FAILED: $job_name (PID: $pid, Exit: $exit_code)"
                    job_statuses[$i]="failed"
                fi
            fi
        elif [ "$job_status" == "completed" ]; then
            echo "✓ COMPLETED: $job_name (PID: $pid)"
        else
            echo "✗ FAILED: $job_name (PID: $pid, Exit: ${job_exit_codes[$i]})"
        fi
    done
    echo "=================="
}

# Function to wait for all jobs
wait_all() {
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for all jobs to complete..."
    
    local total_jobs=${#pids[@]}
    local completed_jobs=0
    
    for i in "${!pids[@]}"; do
        local pid=${pids[$i]}
        local job_name=${job_names[$i]}
        local job_status=${job_statuses[$i]}
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for: $job_name (PID: $pid)"
        wait "$pid"
        local exit_code=$?
        job_exit_codes[$i]=$exit_code  # Update the exit code
        completed_jobs=$((completed_jobs + 1))
        
        if [ $exit_code -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ [$completed_jobs/$total_jobs] Completed: $job_name"
            job_statuses[$i]="completed"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ [$completed_jobs/$total_jobs] Failed: $job_name (exit code: $exit_code)"
            job_statuses[$i]="failed"
        fi
    done
    
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] All jobs completed!"
    echo "Log directory: $log_dir"
}

# Function to kill all jobs
kill_all() {
    echo ""
    echo "Killing all running jobs..."
    
    for i in "${!pids[@]}"; do
        local pid=${pids[$i]}
        local job_name=${job_names[$i]}
        
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing: $job_name (PID: $pid)"
            kill "$pid"
        fi
    done
    
    # Wait a bit then force kill if necessary
    sleep 3
    
    for i in "${!pids[@]}"; do
        local pid=${pids[$i]}
        local job_name=${job_names[$i]}
        
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing: $job_name (PID: $pid)"
            kill -9 "$pid"
        fi
    done
}

# Trap to handle Ctrl+C
trap 'echo ""; echo "Interrupted! Killing all jobs..."; kill_all; exit 1' INT

# Main execution
echo "=== CelebA-HQ Parallel Training Script ==="
echo "Timestamp: $timestamp"
echo "Log directory: $log_dir"
echo "Configuration: n_iter=$n_iter, batch_size=$batch_size"
echo ""

# Start all jobs in parallel
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting all jobs in parallel..."

run_parallel "arcface" "dodging" "cuda:0"
run_parallel "arcface" "impersonation" "cuda:1"
run_parallel "cosface" "dodging" "cuda:2"
run_parallel "cosface" "impersonation" "cuda:3"
run_parallel "facenet" "dodging" "cuda:4"
run_parallel "facenet" "impersonation" "cuda:5"
run_parallel "mobileface" "dodging" "cuda:6"
run_parallel "mobileface" "impersonation" "cuda:7"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All jobs submitted! Total: ${#pids[@]} jobs"
echo ""
echo "Commands to monitor progress:"
echo "  - Show status: kill -USR1 $$"
echo "  - Kill all jobs: kill -USR2 $$"
echo "  - Check logs: tail -f $log_dir/*.log"
echo ""

# Set up signal handlers for monitoring
trap 'show_status' USR1
trap 'kill_all; exit 1' USR2

# Show initial status
show_status

# Wait for all jobs to complete
wait_all

echo "=== All CelebA-HQ training jobs completed! ==="