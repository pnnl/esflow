# AGENTS.md

AWS CDK (Python) app deploying a single stack: an ECS Fargate service (image
from ECR `esflow` repo) behind an internal ALB, in the existing `esflow` VPC.
One stack, one file: `esflow_v2_infra/esflow_v2_infra_stack.py`. `app.py` just
instantiates it with a pinned account/region.

## Setup
```
source .venv/bin/activate   # venv already created (Python 3.14 via pyenv)
pip install -r requirements.txt -r requirements-dev.txt
```

## Commands
- `cdk synth` — synthesize CloudFormation (also validates the stack).
- `cdk diff` — compare against deployed stack before deploying.
- `cdk deploy` — deploy `EsflowV2InfraStack`.
- `pytest` — run tests in `tests/unit/`. The one existing test only asserts the
  stack synthesizes without error; it has no real assertions (commented out
  example from the CDK template).

`cdk` is a global npm CLI (not in requirements.txt); it drives `python3 app.py`
per `cdk.json`.

## Key facts / gotchas
- `env=cdk.Environment(account="793452511035", region="us-west-2")` in
  `app.py` is required and intentional — the stack imports existing resources
  (VPC, subnets, security group, ECR repo, secret) by ID/ARN, so it must
  synth/deploy against that exact account/region or the imports won't resolve.
- All networking (VPC, subnets, security group) and the ECR repo/secret are
  **imported**, not created, via hardcoded IDs/ARNs at the top of
  `esflow_v2_infra_stack.py`. Update those constants if the underlying AWS
  resources change; don't try to create new VPC/SG resources here.
- The ALB and ECS tasks share one existing security group
  (`SHARED_SECURITY_GROUP_ID`); the stack only adds an HTTP ingress rule to it
  for specific PNNL CIDR ranges. The ALB itself is internal
  (`internet_facing=False`).
- Container expects `AI_INCUBATOR_KEY` from Secrets Manager and listens on
  port 8000 with a `/health` health check path; ALB listens on port 80.
- `cdk.json` context has AWS CDK v2 feature flags pre-set for a new-ish CDK
  version (aws-cdk-lib pinned `>=2.265.0,<3.0.0`) — don't casually strip/reset
  these when regenerating `cdk.json`.
- `cdk.out/` is a build artifact (gitignored); don't hand-edit it.
</content>
