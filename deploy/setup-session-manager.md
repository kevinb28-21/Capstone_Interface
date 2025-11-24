# Setting Up AWS Systems Manager Session Manager

This guide will help you set up Session Manager so you can connect to EC2 without SSH.

## Prerequisites

1. SSM Agent (usually pre-installed on Amazon Linux 2 and Ubuntu 18.04+)
2. IAM Role with SSM permissions attached to the EC2 instance
3. Security Group allowing outbound HTTPS (port 443) to AWS endpoints

## Step 1: Check if SSM Agent is Installed

Connect via SSH (if possible) or use EC2 Instance Connect, then run:

```bash
# Check if SSM agent is running
sudo systemctl status amazon-ssm-agent

# If not installed, install it:
# For Ubuntu:
sudo snap install amazon-ssm-agent --classic
sudo systemctl enable amazon-ssm-agent
sudo systemctl start amazon-ssm-agent

# For Amazon Linux 2:
sudo yum install -y amazon-ssm-agent
sudo systemctl enable amazon-ssm-agent
sudo systemctl start amazon-ssm-agent
```

## Step 2: Create IAM Role for SSM

1. Go to **IAM Console** → **Roles** → **Create Role**
2. Select **AWS Service** → **EC2** → **Next**
3. Search for and select **AmazonSSMManagedInstanceCore** policy
4. Click **Next** → Name it `EC2-SSM-Role` → **Create Role**

## Step 3: Attach IAM Role to EC2 Instance

1. Go to **EC2 Console** → Select your instance
2. Click **Actions** → **Security** → **Modify IAM role**
3. Select the `EC2-SSM-Role` you just created
4. Click **Update IAM role**

## Step 4: Verify SSM Agent is Running

Wait 1-2 minutes for the role to propagate, then:

1. Go to **Systems Manager Console** → **Fleet Manager**
2. Your instance should appear in the list
3. If it shows "Online", Session Manager is ready

## Step 5: Connect via Session Manager

1. **EC2 Console** → Select your instance → **Connect**
2. Choose **Session Manager** tab
3. Click **Connect**

You'll get a browser-based terminal!

## Troubleshooting

### Instance not showing in Fleet Manager:
- Wait 5-10 minutes after attaching IAM role
- Check SSM agent is running: `sudo systemctl status amazon-ssm-agent`
- Check agent logs: `sudo tail -f /var/log/amazon/ssm/amazon-ssm-agent.log`

### SSM Agent not starting:
```bash
# Restart the agent
sudo systemctl restart amazon-ssm-agent
sudo systemctl status amazon-ssm-agent
```

### IAM Role not working:
- Verify the role is attached: EC2 → Instance → Security tab
- Check the role has `AmazonSSMManagedInstanceCore` policy
- Ensure instance can reach AWS endpoints (check security group outbound rules)

