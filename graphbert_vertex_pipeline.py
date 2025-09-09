from google_cloud_pipeline_components import aiplatform as gcc_aip
from kfp import dsl

# 你的镜像地址（Artifact Registry）
IMAGE = "us-central1-docker.pkg.dev/<PROJECT_ID>/<REPO>/graphbert-pipeline:latest"

def make_step_op(step_name: str, step: str, workdir: str):
    return dsl.ContainerOp(
        name=step_name,
        image=IMAGE,
        command=["python", "/app/pipeline_runner.py"],
        arguments=["--step", step, "--workdir", workdir]
    )

@dsl.pipeline(
    name="graphbert-vertex-pipeline",
    description="GraphBert training end-to-end on Vertex AI"
)
def graphbert_pipeline(workdir: str = "gs://<YOUR_BUCKET>/graphbert"):
    step1 = make_step_op("step1-data", "step1", workdir)
    step2 = make_step_op("step2-embed", "step2", workdir).after(step1)
    step3 = make_step_op("step3-config", "step3", workdir).after(step2)
    pretrain = make_step_op("pretrain", "pretrain", workdir).after(step3)
    finetune = make_step_op("finetune", "finetune", workdir).after(pretrain)
