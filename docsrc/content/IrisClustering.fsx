(*** hide ***)
// This block of code is omitted in the generated HTML documentation. Use 
// it to define helpers that you do not want to show in the documentation.
//#r "../../packages/formatting/FSharp.Plotly/lib/netstandard2.0/Fsharp.Plotly.dll"
#r "netstandard"
#r "../../lib/Formatting/FSharp.Plotly.dll"
open FSharp.Plotly
(**
Sample: Clustering iris data set
===============================

**)

#load "../../FSharpML.fsx"


open System;
open Microsoft.ML
open Microsoft.ML.Data;
open FSharpML
open FSharpML.EstimatorModel


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


// (Optional) Peek data 
(*** define-output: plot1 ***)
mlContext.CreateEnumerable<IrisData>(fullData,false)
|> Seq.groupBy (fun items -> items.Label)
|> Seq.map (fun (k,values) -> 
    let x = values |> Seq.map (fun items -> items.SepalLength) 
    let y = values |> Seq.map (fun items -> items.SepalWidth) 
    Chart.Point(x,y,Name=sprintf "Label: %.0f" k)
    )
|> Chart.Combine
(*** include-it: plot1 ***)



//Split dataset in two parts: TrainingDataset (80%) and TestDataset (20%)
let struct(trainingDataView, testingDataView) = mlContext.Clustering.TrainTestSplit(fullData, testFraction = 0.2)



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

// STEP4: Evaluate accuracy of the model
let metrics = 
    model
    |> Evaluation.Clustering.InitEvaluate(Score=DefaultColumnNames.Score, Features=DefaultColumnNames.Features) testingDataView

