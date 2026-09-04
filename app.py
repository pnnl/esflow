#!/usr/bin/env python3
import os

import aws_cdk as cdk

from esflow_v2_infra.esflow_v2_infra_stack import EsflowV2InfraStack


app = cdk.App()
EsflowV2InfraStack(app, "EsflowV2InfraStack",
    # Pinned to the account/region that owns the esflow VPC, ECR repo, and
    # AI_INCUBATOR_KEY secret this stack imports/references. Required for the
    # ec2.Vpc.from_lookup() context lookup to resolve at synth time.
    env=cdk.Environment(account="793452511035", region="us-west-2"),
    )

app.synth()
