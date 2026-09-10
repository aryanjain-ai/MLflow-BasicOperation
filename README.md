# MLflow MLOps Learning Project

This project demonstrates the fundamentals of experiment tracking, model versioning, remote MLflow tracking, and cloud-based MLflow infrastructure.

The project currently uses:

- Python
- Scikit-learn
- MLflow
- DagsHub
- Git & GitHub
- AWS CLI
- AWS IAM
- Amazon S3
- Conda

The goal is to understand how an ML experiment moves from a local development environment toward a reproducible, remotely tracked MLOps workflow.

---

# 1. Project Environment

A separate Conda environment is used so that the Python packages required by this project do not interfere with packages used by other projects.

Create the environment:

```bash
conda create -n mlproj python=3.10
```

Activate it:

```bash
conda activate mlproj
```

### Explanation

`conda create` creates an isolated Python environment.

`-n mlproj` gives the environment the name `mlproj`.

`python=3.10` specifies the Python version used by the project.

`conda activate mlproj` tells the terminal to use the Python interpreter and packages installed inside this environment.

---

# 2. Install Project Dependencies

Install the dependencies listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

### Explanation

`python` selects the Python interpreter from the active environment.

`-m pip` runs the `pip` package installer associated with that Python interpreter.

`install` tells pip to install packages.

`-r requirements.txt` tells pip to read the packages required by the project from `requirements.txt`.

This makes it easier for another developer to recreate the project's software environment.

---

# 3. Dependency Compatibility

This project uses an older MLflow version:

```text
mlflow==2.2.2
```

During setup, a compatibility problem occurred between the older MLflow/package stack and newer Python packaging dependencies.

A compatible setuptools version was installed:

```bash
python -m pip install setuptools==65.5.1
```

### What I learned

Installing one package can automatically install many other packages that it depends on.

For example:

```text
My Project
    |
    v
MLflow
    |
    +---- dependency A
    +---- dependency B
    +---- dependency C
```

These packages are called **dependencies**.

Different versions of packages may not always work together.

Dependency management therefore involves making sure the versions of all required packages are compatible.

---

# 4. Machine Learning Experiment

The experiment trains an ElasticNet regression model on the Wine Quality dataset.

Important imports include:

```python
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

### Explanation

- `pandas` is used to load and manipulate the dataset.
- `numpy` provides numerical operations.
- `scikit-learn` provides the ElasticNet model and evaluation metrics.
- `mlflow` tracks experiments, parameters, metrics, and models.

Some Python modules such as:

```python
import os
import sys
import logging
import warnings
```

are part of Python's standard library and therefore do not need to be installed separately.

---

# 5. Model Evaluation

The project evaluates the model using:

```python
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    return rmse, mae, r2
```

The metrics are:

- RMSE — Root Mean Squared Error
- MAE — Mean Absolute Error
- R² — coefficient of determination

These metrics allow different model runs to be compared.

---

# 6. Hyperparameters

ElasticNet uses two important hyperparameters:

```python
alpha
l1_ratio
```

The program allows these values to be supplied from the command line.

For example:

```bash
python example.py 0.2 0.8
```

Inside Python:

```python
alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
```

### Explanation

`sys.argv` contains the arguments supplied when the Python program is started.

For:

```bash
python example.py 0.2 0.8
```

Python receives approximately:

```text
sys.argv[0] = example.py
sys.argv[1] = 0.2
sys.argv[2] = 0.8
```

Therefore:

```text
alpha = 0.2
l1_ratio = 0.8
```

This allows multiple experiments to be run without changing the Python source code every time.

---

# 7. Training the Model

The ElasticNet model is created and trained with:

```python
lr = ElasticNet(
    alpha=alpha,
    l1_ratio=l1_ratio,
    random_state=42
)

lr.fit(train_x, train_y)
```

### Explanation

Creating `ElasticNet(...)` creates the model object.

Calling:

```python
lr.fit(...)
```

actually trains the model using the training data.

---

# 8. MLflow Experiment Tracking

MLflow runs are used to track each experiment.

```python
with mlflow.start_run():
```

Parameters are logged:

```python
mlflow.log_param("alpha", alpha)
mlflow.log_param("l1_ratio", l1_ratio)
```

Metrics are logged:

```python
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("r2", r2)
mlflow.log_metric("mae", mae)
```

### Why MLflow?

Without experiment tracking, it becomes difficult to remember:

```text
Which parameters did I use?

What RMSE did that experiment produce?

Which model belonged to that experiment?

Was the previous model better?
```

MLflow records this information.

The basic structure is:

```text
MLflow Experiment
       |
       +---- Run 1
       |      alpha = 0.5
       |      l1_ratio = 0.5
       |      RMSE = ...
       |
       +---- Run 2
              alpha = 0.2
              l1_ratio = 0.8
              RMSE = ...
```

---

# 9. MLflow Model Logging

The trained model is logged using MLflow:

```python
mlflow.sklearn.log_model(
    lr,
    "model",
    registered_model_name="ElasticnetWineModel",
    signature=signature
)
```

### Explanation

Scikit-learn performs the actual model training.

MLflow then stores information about the trained model and allows the model to be tracked and versioned.

For example:

```text
ElasticnetWineModel
       |
       +---- Version 1
       |
       +---- Version 2
       |
       +---- Version 3
```

A new experiment does not necessarily create a completely different registered model.

It can create a **new version of the same registered model**.

---

# 10. Local MLflow

MLflow can run locally.

Start the local MLflow UI with:

```bash
mlflow ui
```

The UI is normally available at:

```text
http://127.0.0.1:5000
```

`127.0.0.1` refers to the local computer.

Therefore:

```text
Browser
   |
   v
127.0.0.1:5000
   |
   v
MLflow running on my Mac
```

Local MLflow data is separate from experiments stored on a remote MLflow server.

---

# 11. DagsHub Remote MLflow Tracking

DagsHub was added to provide hosted remote MLflow tracking.

Install DagsHub:

```bash
python -m pip install dagshub
```

The project initializes DagsHub using:

```python
import dagshub

dagshub.init(
    repo_owner="aryanjain-ai",
    repo_name="MLflow-BasicOperation",
    mlflow=True
)
```

### Explanation

Previously:

```text
Python Program
      |
      v
Local MLflow
      |
      v
Local experiment data
```

With DagsHub:

```text
Python Program
      |
      v
MLflow Client
      |
      | Internet
      v
DagsHub-hosted MLflow
      |
      +---- Experiments
      +---- Parameters
      +---- Metrics
      +---- Models
      +---- Model Versions
```

MLflow is the experiment tracking software.

DagsHub provides infrastructure that hosts a remote MLflow service.

---

# 12. Git and GitHub

Git is used to version the project's source code.

Important workflow:

```text
Working Directory
       |
       | git add
       v
Staging Area
       |
       | git commit
       v
Local Git Repository
       |
       | git push
       v
GitHub
```

Typical commands:

```bash
git status
git add .
git commit -m "description of change"
git push origin main
```

### Explanation

`git status`

Shows which files have changed.

`git add`

Selects changes for the next commit.

`git commit`

Creates a versioned snapshot locally.

`git push`

Uploads local commits to GitHub.

GitHub stores the project's source code.

MLflow stores information about what happened when the ML experiments were executed.

A useful distinction is:

```text
GitHub
"What code did I write?"

MLflow
"What happened when I ran the experiment?"
```

---

# 13. AWS Setup

The next stage moves the MLflow infrastructure to AWS.

The goal is approximately:

```text
My Mac
Training Code
     |
     | Internet
     v
AWS EC2
     |
MLflow Tracking Server
     |
     v
Amazon S3
     |
Models / Artifacts
```

---

# 14. AWS IAM

An IAM user was created for programmatic AWS access.

IAM stands for:

```text
Identity and Access Management
```

A simple mental model is:

```text
AWS Account
    =
Building

IAM User
    =
Employee

Permissions
    =
What the employee is allowed to do

Credentials
    =
How software proves which employee it is
```

The AWS credentials must NEVER be committed to GitHub.

Never put the following in this repository:

```text
AWS Access Key ID
AWS Secret Access Key
Passwords
API tokens
Private credentials
```

---

# 15. AWS CLI

AWS CLI allows AWS to be controlled from the terminal.

It was installed on macOS using the official AWS installation script.

Verify installation:

```bash
aws --version
```

### PATH Issue Encountered

AWS CLI was installed under:

```text
~/.local/bin
```

but zsh initially could not find the `aws` command.

The following was added to `~/.zshrc`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Then the configuration was reloaded:

```bash
source ~/.zshrc
```

### Explanation

`PATH` is the list of directories that the shell searches when trying to find programs.

Originally:

```text
Type: aws
     |
     v
zsh searches PATH
     |
     X aws not found
```

After adding `~/.local/bin`:

```text
Type: aws
     |
     v
zsh searches PATH
     |
     v
~/.local/bin/aws
     |
     v
AWS CLI runs
```

`.zshrc` is a startup configuration file for the zsh shell.

Adding the PATH setting to `.zshrc` makes the setting available when new zsh sessions start.

Running:

```bash
source ~/.zshrc
```

reloads the configuration immediately without having to close and reopen Terminal.

---

# 16. Configure AWS CLI

AWS CLI was configured using:

```bash
aws configure
```

It asks for:

```text
AWS Access Key ID
AWS Secret Access Key
Default region
Default output format
```

The project uses:

```text
Region: us-east-1
Output format: json
```

`us-east-1` corresponds to the AWS N. Virginia region.

### Important

Credentials are entered locally.

They must NOT be written in this README or committed to Git.

Authentication can be tested with:

```bash
aws sts get-caller-identity
```

---

# 17. AWS Regions

AWS infrastructure exists in multiple geographic regions.

Examples:

```text
N. Virginia    us-east-1
Ohio           us-east-2
Oregon         us-west-2
```

Resources can belong to particular regions.

Therefore, if an EC2 instance is created in:

```text
us-east-1
```

but the AWS Console is displaying:

```text
us-east-2
```

the instance may appear to be missing even though it exists in another region.

For this project, AWS resources should be kept in the same intended region whenever practical.

---

# 18. Amazon S3

An Amazon S3 bucket was created for MLflow artifact storage.

Current bucket:

```text
mlflow-buc20
```

S3 stands for **Simple Storage Service**.

The purpose of S3 in this architecture is to provide cloud object storage for MLflow artifacts such as trained model files.

Architecture:

```text
MLflow Tracking Server
        |
        v
Amazon S3
        |
        +---- Model artifacts
        +---- Other experiment artifacts
```

The bucket should remain private unless there is a specific reason for public access.

Authorized AWS identities can access a private S3 bucket through IAM permissions.

---

# 19. Upcoming AWS EC2 Setup

The next stage is to create an EC2 instance.

EC2 provides a virtual computer running in AWS.

The planned architecture is:

```text
LOCAL COMPUTER
example.py
     |
     | sends MLflow requests
     v
--------------------------------
AWS
--------------------------------
     |
     v
EC2 Virtual Machine
     |
MLflow Tracking Server
     |
     v
S3 Bucket
     |
Model / Artifact Storage
```

The MLflow tracking server will listen for requests from the training application.

The tutorial uses port:

```text
5000
```

The exact EC2 and security-group configuration should be reviewed before exposing the server to the internet.

---

# 20. Security Notes

This project is being developed as a learning project.

Important security practices:

- Never commit AWS credentials.
- Never put secrets in README files.
- Keep S3 buckets private unless public access is specifically required.
- Prefer least-privilege IAM permissions for production systems.
- Avoid using root AWS credentials for applications.
- Rotate credentials if they are accidentally exposed.
- Remove unused AWS resources after experiments to avoid unnecessary costs.

The tutorial may use broader permissions for simplicity. Production systems should use more restrictive IAM policies.

---

# 21. Current Architecture

At this stage, the concepts covered are:

```text
                    CODE
                     |
                     v
                Git / GitHub
                     |
                     |
              Python Training
                     |
                     v
                   MLflow
                  /      \
                 /        \
        Experiment          Model
         Tracking          Registry
             |
             v
     Remote MLflow/DagsHub


AWS learning path:

Python Training
      |
      v
MLflow Client
      |
      v
EC2 MLflow Server
      |
      v
S3 Artifact Storage
```

---

# 22. Key Concepts Learned

### Environment

An isolated place containing the Python interpreter and packages needed by the project.

### Package

Reusable software installed into the Python environment.

### Dependency

A package that another piece of software relies on.

### Version

A particular release of a software package.

### Dependency Management

Making sure required package versions work together.

### MLflow

Software for tracking ML experiments, models, parameters, metrics, and artifacts.

### DagsHub

A service that can host remote MLflow infrastructure.

### Git

A version-control system for tracking changes to source code.

### GitHub

A remote platform for hosting Git repositories and collaborating on code.

### AWS CLI

A command-line program that sends commands to AWS.

### IAM

AWS's system for identities and permissions.

### S3

AWS object storage used here for MLflow artifacts.

### EC2

AWS virtual machines. The project will use EC2 to host the MLflow tracking server.

### PATH

A list of directories that the shell searches when looking for executable programs.

### zsh

The shell used by the macOS Terminal to interpret commands.

### `.zshrc`

A configuration file read by zsh when starting an interactive shell.

---

# 23. Project Status

Completed:

- [x] Create Python environment
- [x] Install MLflow
- [x] Resolve dependency compatibility
- [x] Train ElasticNet model
- [x] Track MLflow experiments locally
- [x] Log parameters and metrics
- [x] Register MLflow model versions
- [x] Set up Git/GitHub repository
- [x] Configure DagsHub remote MLflow tracking
- [x] Create AWS account
- [x] Configure IAM identity
- [x] Install AWS CLI
- [x] Configure AWS CLI
- [x] Create S3 artifact bucket
- [ ] Create EC2 instance
- [ ] Configure EC2 security
- [ ] Install MLflow on EC2
- [ ] Connect MLflow server to S3
- [ ] Connect local training code to AWS MLflow server
- [ ] Add DVC
- [ ] Add Docker
- [ ] Add CI/CD
- [ ] Deploy application/model
- [ ] Add monitoring

---

# 24. Reproducibility Goal

Eventually, another developer should be able to:

```text
Clone repository
      |
      v
Create environment
      |
      v
Install requirements
      |
      v
Configure required external services
      |
      v
Run experiment
      |
      v
Reproduce the ML workflow
```

This is one of the central goals of MLOps: making machine-learning systems reproducible, trackable, deployable, and maintainable.

### Amazon S3

Created an S3 bucket to store MLflow model artifacts:

```text
mlflow-buc20

# Remote MLflow Tracking Server on AWS EC2

## Overview

This project extends a local MLflow experiment-tracking setup into a remote AWS-based MLflow architecture.

Initially, MLflow ran entirely on my local machine:

Local Machine
    |
    |-- Model Training
    |-- MLflow Tracking
    |-- Local Artifact Storage
    |
    +--> MLflow UI: http://127.0.0.1:5000

The goal of the AWS setup is to separate model training from the MLflow tracking infrastructure.

Target architecture:

Local Machine
      |
      | HTTP / MLflow Tracking Requests
      v
Internet
      |
      v
AWS Security Group
      |
      | TCP Port 5000
      v
AWS EC2 Instance
      |
      |-- MLflow Tracking Server
      |-- SQLite Backend Store
      |
      +------> Amazon S3
                |
                +--> MLflow Artifacts

This allows experiments to eventually be executed from another machine while using a centralized MLflow tracking server hosted on AWS.

---

# 1. AWS EC2

Amazon EC2 provides the remote Linux machine used to host the MLflow tracking server.

After connecting to the Ubuntu EC2 instance, the shell prompt looks similar to:

    ubuntu@ip-172-31-23-111:~$

This is important because commands executed here are running on the remote EC2 machine, not on the local Mac.

The EC2 instance has multiple relevant network addresses.

Example:

    127.0.0.1
    172.31.x.x
    52.206.x.x

These addresses have different purposes.

### 127.0.0.1 — Loopback / Localhost

127.0.0.1 is a special loopback address.

It always means:

    "this same computer"

Therefore, when this command is executed inside EC2:

    curl http://127.0.0.1:5000

127.0.0.1 refers to the EC2 machine itself.

If the same address is used on a Mac, it refers to the Mac instead.

Therefore:

    EC2 -> 127.0.0.1 -> EC2 itself

while:

    Mac -> 127.0.0.1 -> Mac itself

127.0.0.1 is not specifically an AWS address.

Every computer can use the loopback address to communicate with services running on itself.

### Private IP

The EC2 instance also receives a private IP, for example:

    172.31.x.x

This address is primarily used for communication within the AWS private network/VPC.

### Public IP

The EC2 instance also has a public IPv4 address, for example:

    52.206.x.x

This allows machines outside AWS, such as a local laptop, to reach the EC2 instance over the internet when the network/security configuration permits it.

The public IP may change after stopping and restarting an EC2 instance unless a static address such as an Elastic IP is configured.

---

# 2. Creating the MLflow Working Directory

A dedicated directory was created on EC2 for the MLflow server:

    mkdir ~/mlflow
    cd ~/mlflow

The "~" represents the current user's home directory.

For the Ubuntu EC2 user:

    ~/mlflow

corresponds approximately to:

    /home/ubuntu/mlflow

---

# 3. Python Virtual Environment

A Python virtual environment was created for the EC2 MLflow installation:

    python3 -m venv .venv

The environment is activated with:

    source .venv/bin/activate

After activation, the terminal prompt changes to something similar to:

    (.venv) ubuntu@ip-172-31-23-111:~/mlflow$

The virtual environment isolates the Python packages required by MLflow from the operating system's Python installation.

This is especially important on modern Ubuntu systems, where system-level Python package installation should generally be avoided for project dependencies.

---

# 4. Installing MLflow

MLflow was installed inside the EC2 virtual environment.

For example:

    pip install mlflow

The EC2 server is running a modern MLflow version.

The local development environment and remote EC2 environment should be treated as separate environments and do not necessarily need to contain identical package installations during initial infrastructure setup.

---

# 5. Amazon S3 Artifact Storage

An Amazon S3 bucket was created for MLflow artifacts.

Example bucket:

    s3://mlflow-buc20

MLflow separates experiment metadata from larger experiment artifacts.

Conceptually:

MLflow Run
    |
    |-- Metadata
    |     |-- Parameters
    |     |-- Metrics
    |     |-- Run information
    |     +-- Experiment information
    |
    +-- Artifacts
          |-- Models
          |-- Plots
          |-- Output files
          +-- Other run artifacts

The EC2 MLflow server is configured with an S3 artifact root so artifacts can be stored in cloud object storage instead of relying only on the EC2 filesystem.

---

# 6. IAM Role and S3 Permissions

The EC2 instance needs permission to communicate with the S3 bucket.

Instead of storing permanent AWS access keys directly on the server, an IAM role was attached to the EC2 instance.

A least-privilege S3 policy was created for the MLflow bucket and attached through the EC2 IAM role.

Conceptually:

EC2 Instance
      |
      v
EC2 IAM Role
      |
      v
S3 Access Policy
      |
      v
MLflow S3 Bucket

This allows applications running on EC2 to obtain temporary AWS credentials through the instance role.

This is preferable to embedding long-lived AWS access keys in application code or configuration files.

---

# 7. AWS CLI

AWS CLI was installed on the EC2 instance.

Its installation was verified using:

    aws --version

The AWS CLI is separate from MLflow and Python.

It provides a command-line interface for communicating with AWS services.

S3 access from EC2 was tested using:

    aws s3 ls s3://mlflow-buc20

A successful command without an AccessDenied error confirmed that the EC2 IAM role had access to the bucket.

If the bucket is empty, the command may return no object names. This does not mean the command failed.

---

# 8. Starting the MLflow Server

The working MLflow server command was:

    mlflow server \
      --host 0.0.0.0 \
      --port 5000 \
      --workers 1 \
      --allowed-hosts "<EC2_PUBLIC_IP>:5000" \
      --default-artifact-root s3://mlflow-buc20

Each argument controls a different part of the server configuration.

---

## mlflow server

    mlflow server

Starts the MLflow tracking server.

The server exposes an HTTP interface that can receive requests from MLflow clients and also provides the MLflow web UI.

---

## --host 0.0.0.0

    --host 0.0.0.0

This configures the server to listen on the EC2 machine's network interfaces rather than restricting the service to loopback-only access.

This is required when another computer needs to communicate with the MLflow server.

This setting does NOT mean that everyone on the internet is automatically allowed to connect.

AWS Security Groups and MLflow's own security controls still determine what traffic is accepted.

---

## --port 5000

    --port 5000

This tells MLflow to listen on TCP port 5000.

An IP address identifies a machine, while a port identifies a service/application on that machine.

Conceptually:

    <IP ADDRESS>:<PORT>

For example:

    52.206.x.x:5000
    |             |
    |             +--> MLflow service
    |
    +--> EC2 machine

The EC2 machine could simultaneously run other applications on other ports.

For example:

    Port 22    -> SSH
    Port 5000  -> MLflow
    Port 8000  -> Some other application

Therefore, port 5000 tells the operating system which application should receive the incoming connection.

---

# 9. IP Address vs Port

A useful networking rule is:

    IP address = which computer?
    Port       = which application on that computer?

For:

    52.206.x.x:5000

the public IP identifies the EC2 instance and port 5000 identifies the MLflow service running on that instance.

When a request reaches the EC2 operating system, Linux checks which process is listening on the requested port.

For example:

Incoming request
      |
      v
EC2 Linux
      |
      | destination port = 5000
      v
Which process is listening on 5000?
      |
      v
MLflow
      |
      v
Request delivered to MLflow

Linux performs this networking/routing behavior. MLflow does not independently search for the request.

---

# 10. Why `curl http://127.0.0.1:5000` Was Used

During debugging, the following command was executed from inside EC2:

    curl http://127.0.0.1:5000

This was a diagnostic test.

It asks:

    "Can this EC2 machine communicate with a web service
     running on itself at port 5000?"

The flow is:

EC2 Terminal
      |
      | curl 127.0.0.1:5000
      v
Linux Loopback Interface
      |
      | Port 5000
      v
MLflow Server

When MLflow HTML was returned, it proved that:

- The MLflow process was running.
- MLflow was listening on port 5000.
- The EC2 operating system could communicate with MLflow locally.

This test intentionally avoids the external internet path.

Therefore:

    curl 127.0.0.1:5000 works

but:

    Mac -> EC2_PUBLIC_IP:5000 fails

would suggest that MLflow itself is working and the problem is more likely related to external networking, AWS Security Groups, routing, or application-level security.

---

# 11. Why 127.0.0.1 Is Not the EC2 Public IP

127.0.0.1 should not be confused with the EC2 public IP.

127.0.0.1 means:

    "myself"

Its meaning depends on which computer executes the request.

If EC2 executes:

    curl http://127.0.0.1:5000

the request goes to EC2 itself.

If a Mac executes:

    curl http://127.0.0.1:5000

the request goes to the Mac itself.

Therefore:

EC2:
    127.0.0.1:5000
          |
          +--> MLflow running on EC2

Mac:
    127.0.0.1:5000
          |
          +--> Whatever is running on the Mac's port 5000

The loopback address never means:

    "go find my AWS server"

It always means:

    "stay on this machine"

---

# 12. Accessing MLflow From the Local Machine

To access the EC2-hosted MLflow server from a local computer, the EC2 public IP must be used.

For example:

    http://<EC2_PUBLIC_IP>:5000

The request path becomes:

Local Computer
      |
      | HTTP request
      v
Internet
      |
      v
EC2 Public IP
      |
      v
AWS Security Group
      |
      | TCP 5000 allowed
      v
EC2 Linux
      |
      | Port 5000
      v
MLflow Server

This is fundamentally different from:

    http://127.0.0.1:5000

which never leaves the machine on which the request originates.

---

# 13. AWS Security Group

An EC2 Security Group acts as a network firewall around the instance.

Even if MLflow is listening on:

    0.0.0.0:5000

AWS can still prevent outside traffic from reaching the machine.

A Security Group rule can allow TCP traffic to port 5000 from a trusted source IP.

Conceptually:

Local Computer
      |
      v
Internet
      |
      v
AWS Security Group
      |
      | Is this source allowed to reach TCP 5000?
      |
      +---- NO ---> Block request
      |
      +---- YES
             |
             v
            EC2

For development/testing, restricting port 5000 to a known public IP is safer than exposing it to the entire internet.

---

# 14. MLflow `--allowed-hosts`

The AWS Security Group and MLflow `--allowed-hosts` solve different problems.

Example:

    --allowed-hosts "<EC2_PUBLIC_IP>:5000"

MLflow receives an HTTP request containing a Host header similar to:

    Host: <EC2_PUBLIC_IP>:5000

MLflow can validate this value before processing the request.

Conceptually:

Request reaches MLflow
      |
      v
Host: <EC2_PUBLIC_IP>:5000
      |
      v
MLflow allowed-host check
      |
      +---- allowed ----> Process request
      |
      +---- not allowed -> Reject request

This is an application-level security check.

It should NOT be interpreted as:

    "Is this particular laptop/user authorized?"

The Host header describes the host/address targeted by the HTTP request.

The source of the incoming network connection is handled separately by networking/firewall controls such as the AWS Security Group.

---

# 15. Why `:5000` Appears in Multiple Places

Port 5000 appears in several places, but each occurrence has a different purpose.

### Server configuration

    --port 5000

Means:

    "MLflow should listen on port 5000."

### Browser/client URL

    http://<EC2_PUBLIC_IP>:5000

Means:

    "Send my request to port 5000 on this EC2 machine."

### Allowed Host configuration

    --allowed-hosts "<EC2_PUBLIC_IP>:5000"

Means:

    "Accept this Host header when validating incoming HTTP requests."

Therefore:

    --port 5000
          |
          +--> Where MLflow listens

    URL :5000
          |
          +--> Where the client sends the request

    --allowed-hosts "...:5000"
          |
          +--> Host header MLflow accepts

These are related but are not the same setting.

---

# 16. Debugging the Invalid Host Header Error

After external networking was configured, the browser successfully reached MLflow but MLflow returned an error similar to:

    Invalid Host header
    possible DNS rebinding attack detected

This was actually useful diagnostic information.

A browser-generated error such as connection refused can indicate that the request never reached the application.

An MLflow-generated "Invalid Host header" response proves that the request reached MLflow far enough for MLflow's application-level security middleware to inspect it.

The request path was therefore already functioning through:

Local Computer
      |
      v
Internet
      |
      v
AWS Security Group
      |
      v
EC2
      |
      v
MLflow
      |
      X Host validation failed

The server was initially configured with an allowed host that did not match the Host value being received.

The working configuration included the public host and port:

    --allowed-hosts "<EC2_PUBLIC_IP>:5000"

After correcting this configuration, the MLflow web UI successfully loaded from the local machine.

---

# 17. Memory / OOM Problem

During initial server startup, the MLflow process unexpectedly terminated with:

    Killed

The EC2 instance had approximately 1 GB of RAM and no swap.

Memory was inspected using:

    free -h

Linux kernel messages were also inspected, for example:

    sudo dmesg | tail

The kernel logs showed an Out-of-Memory (OOM) event in which a Python process was killed.

The problem was therefore not initially an MLflow networking problem.

It was an operating-system resource problem.

Conceptually:

MLflow starts
      |
      v
Multiple server processes/workers
      |
      v
Memory usage increases
      |
      v
Small EC2 instance runs out of RAM
      |
      v
Linux OOM Killer
      |
      v
Python/MLflow process terminated

The development setup was adjusted to use one worker:

    --workers 1

This reduced memory consumption enough for the MLflow server to remain running.

For a larger production workload, using an appropriately sized instance would be preferable to relying on an extremely memory-constrained server.

---

# 18. SQLite Backend Store

When no explicit backend database was supplied, MLflow reported that it was using:

    sqlite:///mlflow.db

This creates a local SQLite database on the EC2 machine.

Conceptually:

MLflow
   |
   +--> Backend Metadata
   |       |
   |       +--> SQLite / mlflow.db
   |
   +--> Artifact Storage
           |
           +--> S3

SQLite is convenient for learning and small deployments.

A production architecture with multiple users, greater concurrency, stronger durability requirements, or larger workloads would typically use a dedicated database service such as PostgreSQL, potentially through Amazon RDS.

---

# 19. Debugging Strategy

A major lesson from this setup is to debug the system layer-by-layer rather than changing everything simultaneously.

A useful order is:

1. Is the MLflow process running?
2. Is MLflow listening on the expected port?
3. Can EC2 itself reach MLflow?
4. Can the external client reach EC2?
5. Does the AWS Security Group permit the connection?
6. Does MLflow accept the HTTP Host?
7. Does EC2 have permission to access S3?

Example diagnostic:

    curl http://127.0.0.1:5000

If this fails:

    Investigate MLflow/process/port configuration.

If this succeeds but external access fails:

    Investigate networking/Security Group/public IP.

If the external request reaches MLflow but produces:

    Invalid Host header

then:

    Investigate MLflow host validation.

If MLflow unexpectedly prints:

    Killed

then:

    Investigate operating-system resources and memory.

If S3 operations return:

    AccessDenied

then:

    Investigate IAM role/policy configuration.

This isolates the failing layer instead of treating the entire deployment as one system.

---

# 20. Current Development Architecture

The current architecture can be summarized as:

Local Development Machine
          |
          | Internet / HTTP
          v
AWS Security Group
          |
          | TCP 5000
          v
EC2 Ubuntu Instance
          |
          +--> MLflow Tracking Server
          |       |
          |       +--> SQLite Backend
          |
          +--> IAM Role
                  |
                  v
               Amazon S3
                  |
                  +--> MLflow Artifact Storage

---

# 21. Local MLflow vs Remote MLflow

MLflow is installed both locally and remotely.

These are separate installations.

Local MLflow:

    Local Machine
        |
        +--> MLflow
              |
              +--> http://127.0.0.1:5000

Remote MLflow:

    Local Machine
        |
        | Internet
        v
    AWS EC2
        |
        +--> MLflow
              |
              +--> http://<EC2_PUBLIC_IP>:5000

Stopping EC2 does NOT uninstall or remove MLflow from the local machine.

Likewise, running MLflow locally does not automatically communicate with the EC2 MLflow server.

The client must explicitly be configured to use the desired tracking server.

---

# 22. Current Limitation: Server Process Lifetime

The MLflow server is currently being started interactively from an EC2 terminal.

This means the server process can terminate when the terminal/session is closed.

Current behavior:

EC2 Instance Connect Session
          |
          +--> MLflow Server Process
                    |
              session ends
                    |
                    v
             MLflow stops

This is suitable for initial testing but not for a persistent deployment.

A future improvement is to run MLflow as a managed background service, for example using systemd or container-based deployment.

Desired behavior:

EC2 starts
    |
    v
MLflow service starts
    |
    v
Terminal can disconnect
    |
    v
MLflow continues running

---

# 23. Next Step: Remote Experiment Tracking

The infrastructure has been tested sufficiently to show that the MLflow UI can be reached from outside EC2.

The next major milestone is sending an actual MLflow experiment from the development machine to the remote tracking server.

Target flow:

Local Python Training Script
          |
          | MLflow tracking requests
          v
EC2 MLflow Tracking Server
          |
          +--> Experiment metadata
          |
          +--> Artifact storage
                    |
                    v
                   S3

This will demonstrate the main benefit of a remote tracking server:

    Model training and MLflow infrastructure
    do not need to run on the same computer.

---

# Key Networking Concepts Learned

## IP address

Identifies the computer/network destination.

Example:

    <EC2_PUBLIC_IP>

## Port

Identifies the application/service on that computer.

Example:

    5000 -> MLflow

## Loopback

    127.0.0.1

means:

    "this computer"

It does not mean AWS or EC2 specifically.

## Public EC2 IP

Allows external machines to address the EC2 instance over the internet, subject to AWS/network security controls.

## AWS Security Group

Controls whether network traffic is permitted to reach the EC2 instance.

## MLflow allowed hosts

Controls which HTTP Host values MLflow accepts after the request reaches the MLflow application.

---

# Key Commands

Activate the EC2 MLflow environment:

    cd ~/mlflow
    source .venv/bin/activate

Start the current development MLflow server:

    mlflow server \
      --host 0.0.0.0 \
      --port 5000 \
      --workers 1 \
      --allowed-hosts "<EC2_PUBLIC_IP>:5000" \
      --default-artifact-root s3://mlflow-buc20

Test MLflow locally from EC2:

    curl http://127.0.0.1:5000

Check EC2 memory:

    free -h

Inspect recent kernel messages:

    sudo dmesg | tail

Test EC2 access to the S3 bucket:

    aws s3 ls s3://mlflow-buc20

---

# Main Takeaway

The deployment contains several independent layers:

Local Client
    |
    v
Internet
    |
    v
AWS Security Group
    |
    v
EC2 Networking
    |
    v
MLflow Tracking Server
    |
    +--> SQLite Metadata
    |
    +--> IAM
            |
            v
           S3

A failure at one layer does not necessarily mean MLflow itself is broken.

The most useful debugging approach is to isolate each layer and verify it independently.

For example:

    curl 127.0.0.1:5000

tests the MLflow server locally,

while opening:

    http://<EC2_PUBLIC_IP>:5000

from another machine tests the external network path as well.

Understanding this separation between application, operating system, networking, security, identity, and storage is one of the primary lessons from deploying MLflow remotely.