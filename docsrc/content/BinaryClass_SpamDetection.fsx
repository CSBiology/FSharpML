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


/// Describes Iris flower. Used as an input to prediction function.
[<CLIMutable>] 
type IrisData = {
    Label : float32
    SepalLength : float32
    SepalWidth: float32
    PetalLength : float32
    PetalWidth : float32    
} 




//Create the MLContext to share across components for deterministic results
let mlContext = MLContext(seed = Nullable 1) // Seed set to any number so you
                                             // have a deterministic environment

// STEP 1: Common data loading configuration
let fullData = 
    mlContext.Data.ReadFromTextFile((__SOURCE_DIRECTORY__  + "./data/iris-full.txt") ,
        hasHeader = true,
        separatorChar = '\t',
        columns =
            [|
                TextLoader.Column("Label", Nullable DataKind.R4, 0)
                TextLoader.Column("SepalLength", Nullable DataKind.R4, 1)
                TextLoader.Column("SepalWidth", Nullable DataKind.R4, 2)
                TextLoader.Column("PetalLength", Nullable DataKind.R4, 3)
                TextLoader.Column("PetalWidth", Nullable DataKind.R4, 4)
            |]
    )



//STEP 2: Process data, create and train the model 
let model = 
    EstimatorModel.create mlContext
    // Process data transformations in pipeline
    |> EstimatorModel.appendBy (fun mlc -> mlc.Transforms.Concatenate(DefaultColumnNames.Features , "SepalLength", "SepalWidth", "PetalLength", "PetalWidth") )
    // Create the model
    |> EstimatorModel.appendBy (fun mlc -> mlc.Clustering.Trainers.KMeans(featureColumn = DefaultColumnNames.Features, clustersCount = 3) )
    // Train the model
    |> EstimatorModel.fit trainingDataView

// STEP3: Run the prediciton on the test data
let predictions =
    model
    |> TransformerModel.transform testingDataView


(**
2. Evaluate and consume the model
---------------------------------

TransformerModel is used to evaluate the model and make prediction on independant data.

**)

// STEP4: Evaluate accuracy of the model
let metrics = 
    model
    |> Evaluation.Clustering.evaluateWith(Score=DefaultColumnNames.Score, Features=DefaultColumnNames.Features) testingDataView





#load "../../FSharpML.fsx"

open System;
open Microsoft.ML
open Microsoft.ML.Data;
open FSharpML

open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Core.Data
open Microsoft.Data.DataView
[<CLIMutable>]
type SpamInput = 
    {
        LabelText : string
        Message : string
    }

[<CLIMutable>]
type SpamPrediction = 
    {
        PredictedLabel : bool
        Score : float32
        Probability : float32
    }


let classify (p : PredictionEngine<_,_>) x = 
    let prediction = p.Predict({LabelText = ""; Message = x})
    printfn "The message '%s' is %b" x prediction.PredictedLabel


let mlContext = MLContext(seed = Nullable 1)

let trainDataPath  = (__SOURCE_DIRECTORY__  + "./data/SMSSpamCollection.txt")


let data = 
    mlContext.Data.ReadFromTextFile(trainDataPath,
        columns = 
            [|
                TextLoader.Column("LabelText" , Nullable DataKind.Text, 0)
                TextLoader.Column("Message" , Nullable DataKind.Text, 1)
            |],
        hasHeader = false,
        separatorChar = '\t')
    
// Create the estimator which converts the text label to a bool then featurizes the text, and add a linear trainer.
let estimator = 
    EstimatorChain()
        .Append(mlContext.Transforms.Conversion.ValueMap(["ham"; "spam"], [false; true],[| struct ("Label", "LabelText") |]))
        .Append(mlContext.Transforms.Text.FeaturizeText("Features", "Message"))
        .AppendCacheCheckpoint(mlContext)
        .Append(mlContext.BinaryClassification.Trainers.StochasticDualCoordinateAscent("Label", "Features"))
        
// Evaluate the model using cross-validation.
// Cross-validation splits our dataset into 'folds', trains a model on some folds and 
// evaluates it on the remaining fold. We are using 5 folds so we get back 5 sets of scores.
// Let's compute the average AUC, which should be between 0.5 and 1 (higher is better).
//let cvResults = mlContext.BinaryClassification.CrossValidate(data, Estimator.downcastEstimator estimator, numFolds = 5);
//let avgAuc = cvResults |> Seq.map (fun struct (metrics,_,_) -> metrics.Auc) |> Seq.average
//printfn "The AUC is %f" avgAuc
    
// Now let's train a model on the full dataset to help us get better results
let model = estimator.Fit(data)

// The dataset we have is skewed, as there are many more non-spam messages than spam messages.
// While our model is relatively good at detecting the difference, this skewness leads it to always
// say the message is not spam. We deal with this by lowering the threshold of the predictor. In reality,
// it is useful to look at the precision-recall curve to identify the best possible threshold.
let newModel = 
    let lastTransformer = 
        BinaryPredictionTransformer<IPredictorProducing<float32>>(
            mlContext, 
            model.LastTransformer.Model, 
            model.GetOutputSchema(data.Schema), 
            model.LastTransformer.FeatureColumn, 
            threshold = 0.15f, 
            thresholdColumn = DefaultColumnNames.Probability);
    let parts = model |> Seq.toArray
    parts.[parts.Length - 1] <- lastTransformer :> _
    TransformerChain<ITransformer>(parts)


// Create a PredictionFunction from our model 
let predictor = model.CreatePredictionEngine<SpamInput, SpamPrediction>(mlContext);

// Test a few examples
[
    "That's a great idea. It should work."
    "free medicine winner! congratulations"
    "Yes we should meet over the weekend!"
    "you win pills and free entry vouchers"
] 
|> List.iter (classify predictor)


