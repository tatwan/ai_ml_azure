# Create and Explore an Azure Machine Learning Workspace

In this exercise, you will create and explore an Azure Machine Learning workspace.

## Create an Azure Machine Learning workspace

As its name suggests, a workspace is a centralized place to manage all of the Azure ML assets you need to work on a machine learning project.

1. In the [Azure portal](https://portal.azure.com/), create a new **Machine Learning** resource, specifying the following settings:
   - **Subscription**: *Your Azure subscription*
   - **Resource group**: *Create a new resource group*
   - **Workspace name**: *Enter a unique name for your workspace*
   - **Region**: *Select the region closest to you*
   - **Storage account**: *Use the default*
   - **Key vault**: *Use the default*
   - **Application insights**: *Use the default*
   - **Container registry**: None
2. When the workspace and its associated resources have been created, view the workspace in the portal.
3. Launch the Azure Machine Learning Studio by selecting the link in the portal, or navigate to [https://ml.azure.com](https://ml.azure.com/).

## Create compute assets

To learn how to work with GPUs within Azure Machine Learning, you’ll work with a GPU compute cluster. A compute cluster scales down after inactivity and will be more cost-efficient. For control activities that are not very compute demanding, you’ll use a CPU compute instance.

1. In Azure Machine Learning studio, open the **Compute** page.

2. On the

    

   Compute instances

    

   tab, add a new compute instance with the following settings. You’ll use this as a workstation to run code in notebooks.

   - **Compute name**: *enter a unique name*
   - **Location**: *The same location as your workspace*
   - **Virtual machine type**: CPU
   - **Virtual machine size**: Standard_DS3_v2

### Create a compute cluster

> **Important:** To run the exercises, you need to use a NCv3 series compute which features NVIDIA’s Tesla V100 GPU.

For the exercises, you’ll use a low-priority GPU compute cluster to save costs. As the compute cluster is low priority, it can be pre-empted if there are other workload isn’t enough capacity in the region. To check for availability, you can use the following command in Cloud Shell to see which regions currently have availability: `az vm list-skus -s standard_nc6s_v3 --output table`

After picking a region that has availability for the necessary GPU cluster, create the cluster in Azure Machine Learning:

1. On the

    

   Compute clusters

    

   tab, add a new compute cluster with the following settings. You’ll use this to execute code that needs GPUs.

   - **Location**: *The same location as your workspace*
   - **Virtual machine priority**: Low priority
   - **Virtual machine type**: GPU
   - **Virtual machine size**: Standard_NC6S_V3 (you may have to choose **Select from all options** to expand the list of sizes)
   - Select **Next**.
   - **Compute name**: *enter a unique name*
   - **Minimum number of nodes**: Leave at default 0.
   - **Maximum number of nodes**: Select 1 node.
   - Select **Create**.

## Clone the repo

1. In Azure Machine Learning Studio, open the **Notebooks** page.

2. Select **Terminal** or the **Open terminal** icon to open a terminal. Check that the **Compute** is set to your compute instance and that the current path is the **/users/your-user-name** folder. You may have to wait until the compute instance is running.

3. Enter the following command to clone a Git repository containing notebooks, data, and other files to your workspace:

   Code

   ```bash
    git clone https://github.com/MicrosoftLearning/mslearn-deep-learning mslearn-deep-learning
   ```

4. When the command has completed, in the **Notebooks** pane, click **↻** to refresh the view and verify that a new **/users/\*your-user-name\*/mslearn-deep-learning** folder has been created.

5. Close the terminal pane, terminating the session.

> **Tip**: The instructions for the exercises are in the **Instructions** folder. The necessary notebooks, scripts, and artifacts are in the **Allfiles** folder.

## Stop your compute instance when done with the exercises

If you’ve finished exploring Azure Machine Learning for now, you should shut down your compute instance to avoid incurring unnecessary charges in your Azure subscription. The compute cluster automatically scales down to 0 nodes when idle.

1. In Azure Machine Learning studio, on the **Compute** page, select your compute instance.
2. Click **Stop** to stop your compute instance. When it has shut down, its status will change to **Stopped**.

> **Note**: Stopping your compute ensures your subscription won’t be charged for compute resources. You will however be charged a small amount for data storage as long as the Azure Machine Learning workspace exists in your subscription. If you have finished exploring Azure Machine Learning, you can delete the Azure Machine Learning workspace and associated resources. However, if you plan to complete any other labs in this series, you will need to repeat this lab to create the workspace and prepare the environment first.

# Load and preprocess data with RAPIDS

To use GPUs to load and preprocess data, data scientists can work with the RAPIDS framework. More specifically, with the cuDF library. In this exercise, you’ll use cuDF to preprocess data with a GPU cluster in Azure Machine Learning.

## Before you start

If you have not already done so, complete the *[Set-up](https://microsoftlearning.github.io/mslearn-deep-learning/Instructions/00-set-up.html)* exercise to create an Azure Machine Learning workspace, compute instance, compute cluster, and to clone the notebooks required for this exercise.

## Create an environment

1. In Azure Machine Learning studio, view the **Environments** page.

2. In the

    

   Custom environments

    

   tab, create a new environment with the following settings:

   - **Name**: rapids-mlflow

   - **Description**: *Optional*

   - **Choose environment type**: Dockerfile

   - **Dockerfile**: Copy and paste the code below:

     Code

     ```
       FROM rapidsai/rapidsai:21.10-cuda11.0-runtime-ubuntu18.04-py3.7
       RUN apt-get update \
       && apt-get install -y fuse \
       && source activate rapids \
       && pip install azureml-mlflow \
       && pip install azureml-dataprep
     ```

3. After reviewing and creating the environment, Azure Machine Learning will automatically build the environment. You can view its progress in the **Details** tab of the environment.

## Open the notebook

Most of the work will be done by our compute cluster which uses GPUs. To get the data and to submit the RAPIDS job, we will use notebooks supported by the compute instance.

1. In [Azure Machine Learning studio](https://ml.azure.com/), view the **Compute** page for your workspace; and on the **Compute Instances** tab, start your compute instance if it is not already running.
2. Navigate to the **Notebooks** page in the Studio.
3. Browse to the **/users/\*your-user-name\*/mslearn-deep-learning/Allfiles/Labs/01-preprocess-data** folder.
4. Run through all cells of the `01-process-data.ipynb` to submit the Python script which loads and preprocesses the flight data with cuDF using the GPU compute cluster.

> **Tip**: To run a code cell, select the cell you want to run and then use the **▷** button to run it.

## Stop your compute instance when done with the exercises

If you’ve finished exploring Azure Machine Learning for now, you should shut down your compute instance to avoid incurring unnecessary charges in your Azure subscription. The compute cluster automatically scales down to 0 nodes when idle.

1. In Azure Machine Learning studio, on the **Compute** page, select your compute instance.
2. Click **Stop** to stop your compute instance. When it has shut down, its status will change to **Stopped**.

> **Note**: Stopping your compute ensures your subscription won’t be charged for compute resources. You will however be charged a small amount for data storage as long as the Azure Machine Learning workspace exists in your subscription. If you have finished exploring Azure Machine Learning, you can delete the Azure Machine Learning workspace and associated resources. However, if you plan to complete any other labs in this series, you will need to repeat this lab to create the workspace and prepare the environment first.

# Train a PyTorch model with a GPU compute cluster

To train a model with GPUs, data scientists can work with the PyTorch library. In this exercise, you’ll use PyTorch to train a Convolutional Neural Network (CNN) model on the MNIST data with a GPU cluster in Azure Machine Learning.

## Before you start

If you have not already done so, complete the *[Set-up](https://microsoftlearning.github.io/mslearn-deep-learning/Instructions/00-set-up.html)* exercise to create an Azure Machine Learning workspace, compute instance, compute cluster, and to clone the notebooks required for this exercise.

## Open the notebook

Most of the work will be done by our compute cluster which uses GPUs. To get the data and to submit the RAPIDS job, we will use notebooks supported by the compute instance.

1. In [Azure Machine Learning studio](https://ml.azure.com/), view the **Compute** page for your workspace; and on the **Compute Instances** tab, start your compute instance if it is not already running.
2. Navigate to the **Notebooks** page in the Studio.
3. Browse to the **/users/\*your-user-name\*/mslearn-deep-learning/Allfiles/Labs/02-train-model** folder.
4. Run through all cells of the `01-train-model.ipynb` notebook to submit the Python script which trains a PyTorch CNN model on the MNIST dataset using the GPU compute cluster.

> **Tip**: To run a code cell, select the cell you want to run and then use the **▷** button to run it.

## Stop your compute instance when done with the exercises

If you’ve finished exploring Azure Machine Learning for now, you should shut down your compute instance to avoid incurring unnecessary charges in your Azure subscription. The compute cluster automatically scales down to 0 nodes when idle.

1. In Azure Machine Learning studio, on the **Compute** page, select your compute instance.
2. Click **Stop** to stop your compute instance. When it has shut down, its status will change to **Stopped**.

> **Note**: Stopping your compute ensures your subscription won’t be charged for compute resources. You will however be charged a small amount for data storage as long as the Azure Machine Learning workspace exists in your subscription. If you have finished exploring Azure Machine Learning, you can delete the Azure Machine Learning workspace and associated resources. However, if you plan to complete any other labs in this series, you will need to repeat this lab to create the workspace and prepare the environment first.

# Deploy Triton with an ONNX model to a managed online endpoint

To deploy a model to an endpoint in Azure Machine Learning, you can use NVIDIA Triton Inference Server. In this exercise, you’ll register an ONNX model that is already trained to the workspace. Deploying to an endpoint will be easy thanks to Triton’s no-code-deployment option in Azure Machine Learning.

## Before you start

If you have not already done so, complete the *[Set-up](https://microsoftlearning.github.io/mslearn-deep-learning/Instructions/00-set-up.html)* exercise to create an Azure Machine Learning workspace, compute instance, and to clone the notebooks required for this exercise. A compute cluster is not needed for this exercise.

## Register the model

For Triton no-code-deployment, a model needs to be registered to the Azure Machine Learning workspace with format set to Triton. To do this, you’ll use the CLI v2 in the terminal hosted by the compute instance. The model you’ll register is a DenseNet model trained to identify images.

1. In [Azure Machine Learning studio](https://ml.azure.com/), view the **Compute** page for your workspace.

2. On the **Compute Instances** tab, start your compute instance if it is not already running.

3. Navigate to the **Notebooks** page in the Studio.

4. Select **Terminal** or the **Open terminal** icon to open a terminal, and ensure that its **Compute** is set to your compute instance.

5. In the terminal, browse to the

    

   /users/*your-user-name*/mslearn-deep-learning/Allfiles/Labs/03-deploy-model

    

   folder with the following command:

   Code

   ```bash
    cd mslearn-deep-learning/Allfiles/Labs/03-deploy-model
   ```

6. Enter the following command to avoid any conflicts with previous versions, remove any ML CLI extensions (both version 1 and 2) with this command:

   Code

   ```bash
    az extension remove -n azure-cli-ml
    az extension remove -n ml
   ```

7. Install the Azure Machine Learning extension with the following command:

   Code

   ```
    az extension add -n ml -y
   ```

8. To use the CLI to register a model, you need to give access to your subscription. Run the command below and follow the instructions in the output to sign in with the email account that has access to the Azure subscription you’re using for this exercise.

   Code

   ```
    az login
   ```

   The model is stored in the **models** folder (note the folder structure required for Triton).

You’ll also find **create-triton-model.yml** which contains the configuration for registering the model. Explore the contents of this YAML file to learn that the registered model will be named **densenet-onnx-model** and the **model_format** is set to **Triton**.

1. In the terminal, run the following command to register the model:

   Code

   ```
    az ml model create -f create-triton-model.yaml
   ```

2. When the command has completed, go to the **Models** pane to find a new model named **densenet-onnx-model**.

3. Back on the **Notebooks** page, close the terminal pane to terminating the session.

## Create the endpoint

To create a managed online endpoint, you’ll use the Azure Machine Learning Studio. Follow the steps below to deploy the previously registered model.

1. Navigate to the **Models** page, and select the **densenet-onnx-model** to view its details.[![Models page](images/03-01-model-page.png)](https://microsoftlearning.github.io/mslearn-deep-learning/Instructions/media/03-01-model-page.png)

2. In the **Details** tab, click on **Deploy**. Then, select **Deploy to real-time endpoint**.[![Model details](images/03-02-model-deploy.png)](https://microsoftlearning.github.io/mslearn-deep-learning/Instructions/media/03-02-model-deploy.png)

3. In the

    

   Create deployment

    

   pane, create a managed online endpoint with the following settings:

   - **Endpoint**: New
   - **Endpoint name**: *Enter a unique name. Add random numbers to ensure uniqueness.*
   - **Compute type**: Managed
   - **Authentication type** Key
   - **Select model**: densenet-onnx-model
   - Keep all default settings for the **Deployment** and **Environment**.
   - **Virtual machine size**: Choose **Standard_NC6s_v3** if possible to use GPU. Alternatively, choose **Standard_F4s_v2** to use CPU.
   - **Instance count**: 1

4. Wait for the endpoint to be created and the deployment to be completed. This usually takes around 10 minutes.

> Tip! If deployment takes exceptionally long, it may be because the name is not unique. Select the information icon under Provisioning state to go to the Azure portal and get an overview of the deployment of resources. If you see this error, delete the endpoint and deployment, and recreate the endpoint with the Studio.

1. Once deployment is ready, you can find it on the **Endpoints** page of the Azure Machine Learning Studio.
2. In the details overview of your endpoint, go to the **Consume** tab and copy and save the **REST endpoint**.
3. Copy and save the **Primary key** under **Authentication**.
4. Save the endpoint and key for the next part of the exercise, where you’ll invoke the endpoint to get the model’s predictions.

## Invoke the endpoint

After registering and deploying the model, you now have a managed online endpoint that you can use to get real-time predictions. You can integrate the endpoint with any app. To test that it works, you’ll invoke the endpoint from a Python notebook.

1. Go to the **Notebooks** page in the Studio.
2. Navigate to the **/users/\*your-user-name\*/mslearn-deep-learning/Allfiles/Labs/03-deploy-model** folder and open the **01-invoke-endpoint.ipynb** notebook.
3. Read the instructions carefully to update necessary parameters and run all cells.

## Stop your compute instance when done with the exercises

If you’ve finished exploring Azure Machine Learning for now, you should shut down your compute instance to avoid incurring unnecessary charges in your Azure subscription. The compute cluster automatically scales down to 0 nodes when idle.

1. In Azure Machine Learning studio, on the **Compute** page, select your compute instance.
2. Click **Stop** to stop your compute instance. When it has shut down, its status will change to **Stopped**.

> **Note**: Stopping your compute ensures your subscription won’t be charged for compute resources. You will however be charged a small amount for data storage as long as the Azure Machine Learning workspace exists in your subscription. If you have finished exploring Azure Machine Learning, you can delete the Azure Machine Learning workspace and associated resources. However, if you plan to complete any other labs in this series, you will need to repeat this lab to create the workspace and prepare the environment first.