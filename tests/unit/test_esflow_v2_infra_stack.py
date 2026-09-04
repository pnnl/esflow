import aws_cdk as core
import aws_cdk.assertions as assertions

from esflow_v2_infra.esflow_v2_infra_stack import EsflowV2InfraStack

# example tests. To run these tests, uncomment this file along with the example
# resource in esflow_v2_infra/esflow_v2_infra_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = EsflowV2InfraStack(app, "esflow-v2-infra")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
