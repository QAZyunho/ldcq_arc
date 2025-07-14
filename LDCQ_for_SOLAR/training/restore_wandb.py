import wandb
import json
import yaml
import os

# 지금 바로 실행 가능
local_run_path = "./wandb/run-20250702_041952-p4h9c6rd/"

# config 읽기
config_path = os.path.join(local_run_path, "config.yaml")
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        print("Config:", config)
else:
    print("config.yaml이 없습니다.")

# 새 run으로 복원 (지금 실행 가능)
run = wandb.init(
    project="LDCQ_single",
    name="recovered_041952_data",
    config=config,
    tags=["recovered"]
)

# 로그 데이터 복원
with open(events_file, 'r') as f:
    for line in f:
        try:
            event = json.loads(line)
            if '_step' in event:
                metrics = {k: v for k, v in event.items() if not k.startswith('_')}
                if metrics:
                    wandb.log(metrics, step=event['_step'])
        except:
            continue

wandb.finish()
print("복원 완료!")