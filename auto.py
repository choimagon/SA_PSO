from pathlib import Path
import subprocess
import yaml

CONFIG_PATH = "config.yaml"
values = [95, 85, 75]   # 돌릴 숫자들

for _ in range(2):
    for v in values:
        # yaml 저장
        with open(CONFIG_PATH, "w") as f:
            yaml.safe_dump({"value": v}, f)

        print(f"\n===== RUN with value={v} =====")
        # main.py 실행
        subprocess.run(["python", "run_v6.py"])
