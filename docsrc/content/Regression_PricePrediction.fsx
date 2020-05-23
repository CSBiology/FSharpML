(*** hide ***)
// This block of code is omitted in the generated HTML documentation. Use 
// it to define helpers that you do not want to show in the documentation.
#r "../../packages/formatting/Newtonsoft.Json/lib/netstandard2.0/Newtonsoft.Json.dll"
#r "../../packages/formatting/FSharp.Plotly/lib/netstandard2.0/FSharp.Plotly.dll" 
#I "../../"
open FSharp.Plotly
(**
Regression: Price prediction 
===============================



| ML.NET version | API type          | Status                        | App Type    | Data type | Scenario            | ML Task                   | Algorithms                  |
|----------------|-------------------|-------------------------------|-------------|-----------|---------------------|---------------------------|-----------------------------|
| v1.40           | Dynamic API | Up-to-date | Console app | .csv files | Price prediction | Regression | Sdca Regression |

In this introductory sample, you'll see how to use FSharpML on top of [ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet) to predict taxi fares. In the world of machine learning, this type of prediction is known as **regression**.

Problem
-------
This problem is centered around predicting the fare of a taxi trip in New York City. At first glance, it may seem to depend simply on the distance traveled. However, taxi vendors in New York charge varying amounts for other factors such as additional passengers, paying with a credit card instead of cash and so on. This prediction can be used in application for taxi providers to give users and drivers an estimate on ride fares.

To solve this problem, we will build an ML model that takes as inputs: 
* vendor ID
* rate code
* passenger count
* trip time
* trip distance
* payment type

and predicts the fare of the ride.

ML task - Regression
--------------------
The generalized problem of **regression** is to predict some continuous value for given parameters, for example:
* predict a house prise based on number of rooms, location, year built, etc.
* predict a car fuel consumption based on fuel type and car parameters.
* predict a time estimate for fixing an issue based on issue attributes.

The common feature for all those examples is that the parameter we want to predict can take any numeric value in certain range. In other words, this value is represented by `integer` or `float`/`double`, not by `enum` or `boolean` types.

Solution
--------
To solve this problem, first we will build an ML model. Then we will train the model on existing data, evaluate how good it is, and lastly we'll consume the model to predict taxi fares.



1. Build and train the model
----------------------------

FSharpML containing two complementary parts named EstimatorModel and TransformerModel covering the full machine lerarning workflow. In order to build an ML model and fit it to the training data we use EstimatorModel.
The 'fit' function in EstimatorModel applied on training data results into the TransformerModel that represents the trained model able to transform other data of the same shape and is used int the second part to evaluate and consume the model.


Building a model includes: uploading data (`taxi-fare-train.csv` with `TextLoader`), transforming the data so it can be used effectively by an ML algorithm (`StochasticDualCoordinateAscent` in this case):

**)

#load "FSharpML.fsx"


open System
open Microsoft.ML
open Microsoft.ML.Data
open FSharpML
open FSharpML.EstimatorModel
open FSharpML.TransformerModel
//open Microsoft.ML.Transforms.Normalizers
open FSharpML



type TaxiTrip = {
    [<LoadColumn(0)>] VendorId       : string
    [<LoadColumn(1)>] RateCode       : string
    [<LoadColumn(2)>] PassengerCount : float32
    [<LoadColumn(3)>] TripTime       : float32
    [<LoadColumn(4)>] TripDistance   : float32
    [<LoadColumn(5)>] PaymentType    : string
    [<LoadColumn(6)>] FareAmount     : float32
    }


//Create the MLContext to share across components for deterministic results
let mlContext = MLContext(seed = Nullable 1) // Seed set to any number so you
                                             // have a deterministic environment

// STEP 1: Common data loading configuration
let trainingData =     
    __SOURCE_DIRECTORY__  + "./data/taxi-fare-train.csv"
    |> DataModel.fromTextFileWith<TaxiTrip> mlContext ',' true
    //Sample code of removing extreme data like "outliers" for FareAmounts higher than $150 and lower than $1 which can be error-data
    |> DataModel.appendFilterByColumn "FareAmount"  1. 150.
    |> DataModel.toDataview

let testingData = 
    __SOURCE_DIRECTORY__  + "./data/taxi-fare-test.csv"
    |> DataModel.fromTextFileWith<TaxiTrip> mlContext ',' true
    |> DataModel.toDataview


// STEP 2: Common data process configuration with pipeline data transformations
let modelbuilding = 
    EstimatorModel.create mlContext
    |> EstimatorModel.Transforms.copyColumn "Label" "FareAmount"
    |> EstimatorModel.Transforms.by (fun tfc -> tfc.Categorical.OneHotEncoding( "VendorIdEncoded", "VendorId") )
    |> EstimatorModel.transformBy (fun tfc -> tfc.Categorical.OneHotEncoding( "RateCodeEncoded", "RateCode") )
    |> EstimatorModel.transformBy (fun tfc -> tfc.Categorical.OneHotEncoding( "PaymentTypeEncoded", "PaymentType") )
    |> EstimatorModel.transformBy (fun tfc -> tfc.NormalizeMeanVariance( "PassengerCount", "PassengerCount") )
    |> EstimatorModel.transformBy (fun tfc -> tfc.NormalizeMeanVariance( "TripTime", "TripTime") )
    |> EstimatorModel.transformBy (fun tfc -> tfc.NormalizeMeanVariance( "TripDistance", "TripDistance") )
    |> EstimatorModel.Transforms.concatenate DefaultColumnNames.Features
                [|"VendorIdEncoded"; "RateCodeEncoded";  "PaymentTypeEncoded";  "PassengerCount";  "TripTime";  "TripDistance"|]
    |> EstimatorModel.appendCacheCheckpoint
    
    // Set the training algorithm (SDCA Regression algorithm)  
    |> EstimatorModel.appendBy (fun mlc -> 
        mlc.Regression.Trainers.Sdca
            (
                labelColumnName = DefaultColumnNames.Label,
                featureColumnName = DefaultColumnNames.Features
                ) )                                 

// STEP 3: Train the model fitting to the DataSet
let model =
    modelbuilding
    |> EstimatorModel.fit trainingData                             


(**
2. Evaluate and consume the model
---------------------------------

TransformerModel is used to evaluate the model and make prediction on independant data.

**)

// STEP 4: Evaluate the model and show accuracy stats
let predictions = 
    model
    |> TransformerModel.transform testingData

let metrics = 
   Evaluation.Regression.evaluate testingData model
   


//let y,yy =    
//    predictions  
//    |> Data.getColumn<float32> mlContext DefaultColumnNames.Score
//    |> Seq.take 100,
//    predictions  
//    |> Data.getColumn<float32> mlContext DefaultColumnNames.Label
//    |> Seq.take 100

//Chart.Point(y, yy)
//|> Chart.Show


// STEP 5: Consume the model and predict a taxifare sample
let taxiTripSample = 
    {
        VendorId = "VTS"
        RateCode = "1"
        PassengerCount = 1.0f
        TripTime = 1140.0f
        TripDistance = 3.75f
        PaymentType = "CRD"
        FareAmount = 0.0f // To predict. Actual/Observed = 15.5
    }

[<CLIMutable>]
type RegresionResult = {    
    Score : float32
}

// 
let predict = 
    TransformerModel.createPredictionEngine<_,TaxiTrip,RegresionResult> model


predict taxiTripSample




