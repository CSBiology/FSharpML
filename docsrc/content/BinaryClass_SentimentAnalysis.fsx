(*** hide ***)
// This block of code is omitted in the generated HTML documentation. Use 
// it to define helpers that you do not want to show in the documentation.
#r "../../packages/formatting/Newtonsoft.Json/lib/netstandard2.0/Newtonsoft.Json.dll"
#r@"../../packages/formatting/FSharp.Plotly/lib/netstandard2.0/FSharp.Plotly.dll" 
#I "../../"
open FSharp.Plotly
(**
Binary classification: Sentiment Analysis for User Reviews
==========================================================

| ML.NET version | API type          | Status                        | App Type    | Data type | Scenario            | ML Task                   | Algorithms                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v1.40           | Dynamic API | README.md updated | Console app | .tsv files | Sentiment Analysis | Two-class  classification | Linear Classification |


In this introductory sample, you'll see how to use FSharpML on top of [ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet) to predict a sentiment (positive or negative) for customer reviews. In the world of machine learning, this type of prediction is known as **binary classification**.


Problem
-------
This problem is centered around predicting if a customer's review has positive or negative sentiment. We will use wikipedia-detox-datasets (one dataset for training and a second dataset for model's accuracy evaluation) that were processed by humans and each comment has been assigned a sentiment label:

0 - negative
1 - positive

Using those datasets we will build a model that will analyze a string and predict a sentiment value of 0 or 1.

ML task - Binary classification
-------------------------------
The generalized problem of **binary classification** is to classify items into one of two classes classifying items into more than two classes is called multiclass classification.

* predict if an insurance claim is valid or not.
* predict if a plane will be delayed or will arrive on time.
* predict if a face ID (photo) belongs to the owner of a device.

The common feature for all those examples is that the parameter we want to predict can take only one of two values. In other words, this value is represented by `boolean` type.

Solution
--------

To solve this problem, first we will build an ML model. Then we will train the model on existing data, evaluate how good it is, and lastly we'll consume the model to predict a sentiment for new reviews.


1. Build and train the model
----------------------------

FSharpML containing two complementary parts named EstimatorModel and TransformerModel covering the full machine lerarning workflow. In order to build an ML model and fit it to the training data we use EstimatorModel.
The 'fit' function in EstimatorModel applied on training data results into the TransformerModel that represents the trained model able to transform other data of the same shape and is used int the second part to evaluate and consume the model.

Building a model includes: 

* Define the data's schema maped to the datasets to read (`wikipedia-detox-250-line-data.tsv` and `wikipedia-detox-250-line-test.tsv`) with a DataReader

* Create an Estimator and transform the data to numeric vectors so it can be used effectively by an ML algorithm (with `FeaturizeText`)

* Choosing a trainer/learning algorithm (such as `FastTree`) to train the model with. 

**)

#load "../../FSharpML.fsx"


open System;
open Microsoft.ML
open Microsoft.ML.Data;
open FSharpML
open FSharpML.EstimatorModel
open FSharpML.TransformerModel



/// Type representing the text to run sentiment analysis on.
[<CLIMutable>] 
type SentimentIssue = 
    { 
        [<LoadColumn(0)>]
        Label : bool

        [<LoadColumn(1)>]
        Text : string 
    }

/// Result of sentiment prediction.
[<CLIMutable>]
type  SentimentPrediction = 
    { 
        // ColumnName attribute is used to change the column name from
        // its default value, which is the name of the field.
        [<ColumnName("PredictedLabel")>]
        Prediction : bool; 

        // No need to specify ColumnName attribute, because the field
        // name "Probability" is the column name we want.
        Probability : float32; 

        Score : float32 
    }


//Create the MLContext to share across components for deterministic results
let mlContext = MLContext(seed = Nullable 1) // Seed set to any number so you
                                             // have a deterministic environment

// STEP 1: Common data loading configuration
let fullData = 
    __SOURCE_DIRECTORY__  + "./data/wikipedia-detox-250-line-all.tsv"
    |> DataModel.fromTextFileWith<SentimentIssue> mlContext '\t' true 

let trainingData, testingData = 
    fullData
    |> DataModel.trainTestSplit 0.2 
    

    // DefaultColumnNames

//STEP 2: Process data, create and train the model 
let model = 
    EstimatorModel.create mlContext
    // Process data transformations in pipeline
    |> EstimatorModel.appendBy (fun mlc -> mlc.Transforms.Text.FeaturizeText("Features" , "Text"))
    // Create the model
    |> EstimatorModel.appendBy (fun mlc -> mlc.BinaryClassification.Trainers.FastTree(labelColumnName = "Label", featureColumnName = "Features"))
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
    |> Evaluation.BinaryClassification.evaluateWith(Label=DefaultColumnNames.Label,Score=DefaultColumnNames.Score) testingData.Dataview



let sampleStatement = { Label = false; Text = "This is a very rude movie" }
// STEP5: Create prediction engine function related to the loaded trained model
let predict = 
    TransformerModel.createPredictionEngine<_,SentimentIssue,SentimentPrediction> model

// Score
let prediction = predict sampleStatement


