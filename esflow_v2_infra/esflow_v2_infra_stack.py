from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_ec2 as ec2,
    aws_ecr as ecr,
    aws_ecs as ecs,
    aws_elasticloadbalancingv2 as elbv2,
    aws_logs as logs,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct

# Existing resources this stack imports rather than creates. These live in the
# esflow VPC (us-west-2), provisioned outside of this CDK app.
VPC_ID = "vpc-0aa0c915a587f4f6a"
PRIVATE_SUBNET_IDS = [
    "subnet-0e62a70e27b2f968e",  # esflow-Private Subnet A (us-west-2a)
    "subnet-09cd7332dbaf43462",  # esflow-Private Subnet B (us-west-2b)
]
PRIVATE_SUBNET_AZS = ["us-west-2a", "us-west-2b"]
SHARED_SECURITY_GROUP_ID = "sg-0841df8391230e7e3"
ECR_REPOSITORY_NAME = "esflow"
ECR_IMAGE_TAG = "latest"
AI_INCUBATOR_KEY_SECRET_ARN = (
    "arn:aws:secretsmanager:us-west-2:793452511035:secret:AI_INCUBATOR_KEY-WYmQSW"
)

CONTAINER_PORT = 8000
ALB_LISTENER_PORT = 80


class EsflowV2InfraStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # --- Imported networking -------------------------------------------------
        vpc = ec2.Vpc.from_vpc_attributes(
            self,
            "EsflowVpc",
            vpc_id=VPC_ID,
            availability_zones=PRIVATE_SUBNET_AZS,
            private_subnet_ids=PRIVATE_SUBNET_IDS,
        )

        # The default VPC security group PNNL already permits DirectConnect/VPN
        # traffic through. Reused for both the ALB and the ECS task ENIs so
        # ALB<->task and PNNL<->ALB traffic is covered by its existing rules.
        shared_security_group = ec2.SecurityGroup.from_security_group_id(
            self,
            "SharedSecurityGroup",
            security_group_id=SHARED_SECURITY_GROUP_ID,
            mutable=True,
        )

        # Explicit inbound rule for the ALB listener port from the PNNL networks
        # that can already reach this VPC (matches the SG's existing 80/22 rules).
        for cidr in ("130.20.0.0/16", "10.20.0.0/16", "10.100.0.0/16"):
            shared_security_group.add_ingress_rule(
                ec2.Peer.ipv4(cidr),
                ec2.Port.tcp(ALB_LISTENER_PORT),
                f"ALB HTTP from PNNL network {cidr}",
            )

        # --- ECR image -------------------------------------------------------------
        repository = ecr.Repository.from_repository_name(
            self, "EsflowRepository", ECR_REPOSITORY_NAME
        )

        # --- Secrets ---------------------------------------------------------------
        ai_incubator_secret = secretsmanager.Secret.from_secret_complete_arn(
            self, "AiIncubatorKeySecret", AI_INCUBATOR_KEY_SECRET_ARN
        )

        # --- ECS cluster -------------------------------------------------------------
        cluster = ecs.Cluster(
            self,
            "EsflowCluster",
            vpc=vpc,
            container_insights_v2=ecs.ContainerInsights.ENABLED,
        )

        log_group = logs.LogGroup(
            self,
            "EsflowServiceLogGroup",
            log_group_name="/ecs/esflow-v2",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        task_definition = ecs.FargateTaskDefinition(
            self,
            "EsflowTaskDefinition",
            cpu=512,
            memory_limit_mib=1024,
        )

        container = task_definition.add_container(
            "EsflowContainer",
            image=ecs.ContainerImage.from_ecr_repository(repository, ECR_IMAGE_TAG),
            logging=ecs.LogDriver.aws_logs(stream_prefix="esflow", log_group=log_group),
            environment={
                "WEB_AGENT_MODE": "planner_executor",
            },
            secrets={
                "AI_INCUBATOR_KEY": ecs.Secret.from_secrets_manager(
                    ai_incubator_secret, field="AI_INCUBATOR_KEY"
                ),
            },
        )
        container.add_port_mappings(
            ecs.PortMapping(container_port=CONTAINER_PORT, protocol=ecs.Protocol.TCP)
        )

        service = ecs.FargateService(
            self,
            "EsflowService",
            cluster=cluster,
            task_definition=task_definition,
            desired_count=1,
            min_healthy_percent=100,
            vpc_subnets=ec2.SubnetSelection(subnets=vpc.private_subnets),
            security_groups=[shared_security_group],
            assign_public_ip=False,
            health_check_grace_period=Duration.seconds(120),
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
        )

        # --- Internal ALB ------------------------------------------------------------
        load_balancer = elbv2.ApplicationLoadBalancer(
            self,
            "EsflowAlb",
            vpc=vpc,
            internet_facing=False,
            vpc_subnets=ec2.SubnetSelection(subnets=vpc.private_subnets),
            security_group=shared_security_group,
        )

        listener = load_balancer.add_listener(
            "EsflowHttpListener",
            port=ALB_LISTENER_PORT,
            protocol=elbv2.ApplicationProtocol.HTTP,
            open=False,
        )

        listener.add_targets(
            "EsflowTargetGroup",
            port=CONTAINER_PORT,
            protocol=elbv2.ApplicationProtocol.HTTP,
            targets=[service],
            health_check=elbv2.HealthCheck(
                path="/health",
                port=str(CONTAINER_PORT),
                protocol=elbv2.Protocol.HTTP,
                healthy_http_codes="200",
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                healthy_threshold_count=2,
                unhealthy_threshold_count=3,
            ),
        )

        CfnOutput(
            self,
            "AlbDnsName",
            value=load_balancer.load_balancer_dns_name,
            description="Internal ALB DNS name; reach the MCP server at http://<this>/mcp",
        )
