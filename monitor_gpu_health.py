import time
import subprocess
import os
import sys

LOG_FILE = "./gpu_monitoring.log"

def get_gpu_metrics():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits"
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        line = result.stdout.strip()
        parts = [p.strip() for p in line.split(",")]
        return {
            "gpu_util": float(parts[0]),
            "mem_util": float(parts[1]),
            "mem_used_mb": float(parts[2]),
            "mem_total_mb": float(parts[3]),
            "temp_c": float(parts[4]),
            "power_w": float(parts[5])
        }
    except Exception as e:
        return None

def main():
    print("=" * 65)
    print("      QUADTREE-JEPA REAL-TIME GPU HEALTH & EFFICIENCY MONITOR      ")
    print("=" * 65)
    print(f"Logging telemetry to: {LOG_FILE}\n")
    
    with open(LOG_FILE, "w") as f:
        f.write("timestamp,gpu_util_pct,mem_used_mb,temp_c,power_w,status\n")
        
    history_util = []
    
    while True:
        m = get_gpu_metrics()
        if m is not None:
            history_util.append(m['gpu_util'])
            if len(history_util) > 10:
                history_util.pop(0)
                
            avg_util = sum(history_util) / len(history_util)
            
            if m['gpu_util'] >= 70:
                status = "OPTIMAL_HIGH_THROUGHPUT"
            elif m['gpu_util'] >= 40:
                status = "MODERATE_UTILIZATION"
            elif m['mem_used_mb'] > 1000:
                status = "CPU_OR_DATA_BOUND"
            else:
                status = "WARMUP_OR_IDLE"
                
            log_line = f"{time.strftime('%H:%M:%S')},{m['gpu_util']:.1f}%,{m['mem_used_mb']:.0f}MB,{m['temp_c']:.0f}C,{m['power_w']:.1f}W,{status}"
            print(f"[{time.strftime('%H:%M:%S')}] GPU: {m['gpu_util']:5.1f}% | VRAM: {m['mem_used_mb']:4.0f}MB / {m['mem_total_mb']:.0f}MB | Temp: {m['temp_c']:.0f}°C | Power: {m['power_w']:4.1f}W | {status}")
            
            with open(LOG_FILE, "a") as f:
                f.write(f"{log_line}\n")
        else:
            print("[Warning] Failed to query nvidia-smi")
            
        time.sleep(3)

if __name__ == "__main__":
    main()
