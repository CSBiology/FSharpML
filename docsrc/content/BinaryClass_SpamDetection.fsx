(*** hide ***)
// This block of code is omitted in the generated HTML documentation. Use 
// it to define helpers that you do not want to show in the documentation.
//#r "../../packages/formatting/FSharp.Plotly/lib/netstandard2.0/Fsharp.Plotly.dll"
#r "netstandard"
#r "../../lib/Formatting/FSharp.Plotly.dll"
open FSharp.Plotly
(**
Binary classification: Spam Detection for Text Messages
=======================================================

| ML.NET version | API type          | Status                        | App Type    | Data type | Scenario            | ML Task                   | Algorithms                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v0.10          | Dynamic API | Up-to-date | Console app | .tsv files | Spam detection | Two-class classification | SDCA (linear learner) |

In this sample, you'll see how to use FSharpML on top of [ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet) to predict whether a text message is spam. In the world of machine learning, this type of prediction is known as **binary classification**.

Problem
-------

Our goal here is to predict whether a text message is spam (an irrelevant/unwanted message). We will use the [SMS Spam Collection Data Set](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) from UCI, which contains close to 6000 messages that have been classified as being "spam" or "ham" (not spam). We will use this dataset to train a model that can take in new message and predict whether they are spam or not.

This is an example of binary classification, as we are classifying the text messages into one of two categories.

Solution
--------
To solve this problem, first we will build an estimator to define the ML pipeline we want to use. Then we will train this estimator on existing data, evaluate how good it is, and lastly we'll consume the model to predict whether a few examples messages are spam.

![Build -> Train -> Evaluate -> Consume](../shared_content/modelpipeline.png)


1. Build and train the model
----------------------------

FSharpML containing two complementary parts named EstimatorModel and TransformerModel covering the full machine lerarning workflow. In order to build an ML model and fit it to the training data we use EstimatorModel.
The 'fit' function in EstimatorModel applied on training data results into the TransformerModel that represents the trained model able to transform other data of the same shape and is used int the second part to evaluate and consume the model.

To build the estimator we will:

- Define how to read the spam dataset that will be downloaded from https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection. 

- Apply several data transformations:

    - Convert the label ("spam" or "ham") to a boolean ("true" represents spam) so we can use it with a binary classifier. 
    - Featurize the text message into a numeric vector so a machine learning trainer can use it

- Add a trainer (such as `StochasticDualCoordinateAscent`).

**)

#load "../../FSharpML.fsx"


open System;
open Microsoft.ML
open Microsoft.ML.Data;
open FSharpML
open FSharpML.EstimatorModel
open FSharpML.TransformerModel


/// Type representing the Message to run analysis on.
[<CLIMutable>] 
type SpamInput = 
    { 
        [<LoadColumn(0)>] LabelText : string
        [<LoadColumn(1)>] Message : string 
    }




//Create the MLContext to share across components for deterministic results
let mlContext = MLContext(seed = Nullable 1) // Seed set to any number so you
                                             // have a deterministic environment

// STEP 1: Common data loading configuration   
let fullData = 
    __SOURCE_DIRECTORY__  + "./data/SMSSpamCollection.txt"
    |> DataModel.fromTextFileWith<SpamInput> mlContext '\t' false 

let trainingData, testingData = 
    fullData
    |> DataModel.BinaryClassification.trainTestSplit 0.1

//STEP 2: Process data, create and train the model 
let model = 
    EstimatorModel.create mlContext
    // Process data transformations in pipeline
    |> EstimatorModel.appendBy (fun mlc -> mlc.Transforms.Conversion.ValueMap(["ham"; "spam"],[false; true],[| struct (DefaultColumnNames.Label, "LabelText") |]))
    |> EstimatorModel.appendBy (fun mlc -> mlc.Transforms.Text.FeaturizeText(DefaultColumnNames.Features, "Message"))
    |> EstimatorModel.appendCacheCheckpoint
    // Create the model
    |> EstimatorModel.appendBy (fun mlc -> mlc.BinaryClassification.Trainers.StochasticDualCoordinateAscent(DefaultColumnNames.Label, DefaultColumnNames.Features))
    // Train the model
    |> EstimatorModel.fit trainingData.Dataview

// STEP3: Run the prediciton on the test data
let predictions =
    model
    |> TransformerModel.transform testingData.Dataview


(**
2. Evaluate and consume the model
---------------------------------

TransformerModel is used to evaluate the model and make prediction on independant data.

**)

// STEP4: Evaluate accuracy of the model
let metrics = 
    model
    |> Evaluation.BinaryClassification.evaluate testingData.Dataview

metrics.Accuracy





//// STEP5: Create prediction engine function related to the loaded trained model
//let predict = 
//    TransformerModel.createPredictionEngine<_,SpamInput,SpamInput> model

//// Score
//let prediction = predict sampleStatement

//// Test a few examples
//[
//    "That's a great idea. It should work."
//    "free medicine winner! congratulations"
//    "Yes we should meet over the weekend!"
//    "you win pills and free entry vouchers"
//] 
//|> List.iter (classify predictor)


