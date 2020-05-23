(*** hide ***)
// This block of code is omitted in the generated HTML documentation. Use 
// it to define helpers that you do not want to show in the documentation.
#r "../../packages/formatting/Newtonsoft.Json/lib/netstandard2.0/Newtonsoft.Json.dll"
#r "../../packages/formatting/FSharp.Plotly/lib/netstandard2.0/FSharp.Plotly.dll" 
#I "../../"
open FSharp.Plotly
(**
Sample: Clustering iris data set
===============================

**)

#load "FSharpML.fsx"


open System;
open Microsoft.ML
open Microsoft.ML.Data
open FSharpML
open FSharpML.Data
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
    let hasHeader = true
    let separatorChar = '\t'
    let columns =
        [|
            TextLoader.Column("Label", DataKind.Single, 0)
            TextLoader.Column("SepalLength", DataKind.Single, 1)
            TextLoader.Column("SepalWidth", DataKind.Single, 2)
            TextLoader.Column("PetalLength", DataKind.Single, 3)
            TextLoader.Column("PetalWidth", DataKind.Single, 4)
        |]

    __SOURCE_DIRECTORY__  + "./data/iris-full.txt"
    |> Data.loadFromTextFile mlContext separatorChar hasHeader columns    
    |> DataModel.ofDataview<string> mlContext


// (Optional) Peek data 
let plot1 = 
    mlContext.Data.CreateEnumerable<IrisData>(fullData.Dataview,false)
    |> Seq.groupBy (fun items -> items.Label)
    |> Seq.map (fun (k,values) -> 
        let x = values |> Seq.map (fun items -> items.SepalLength) 
        let y = values |> Seq.map (fun items -> items.SepalWidth) 
        Chart.Point(x,y,Name=sprintf "Label: %.0f" k)
        )
    |> Chart.Combine
(*** include-value: plot1 ***)



//Split dataset in two parts: TrainingData (80%) and TestData (20%)
let trainingData, testingData = 
    fullData
    |> DataModel.trainTestSplit 0.2 



//STEP 2: Process data, create and train the model 
let model = 
    EstimatorModel.create mlContext
    // Process data transformations in pipeline
    |> EstimatorModel.appendBy (fun mlc -> mlc.Transforms.Concatenate(DefaultColumnNames.Features , "SepalLength", "SepalWidth", "PetalLength", "PetalWidth") )
    // Create the model
    |> EstimatorModel.appendBy (fun mlc -> 
            mlc.Clustering.Trainers.KMeans(
                        featureColumnName = DefaultColumnNames.Features, 
                        numberOfClusters = 3
                    ) )
    // Train the model
    |> EstimatorModel.fit trainingData.Dataview

// STEP3: Run the prediciton on the test data
let predictions =
    model
    |> TransformerModel.transform testingData.Dataview

// STEP4: Evaluate accuracy of the model
let metrics = 
    model
    |> Evaluation.Clustering.evaluateWith(Score=DefaultColumnNames.Score, Features=DefaultColumnNames.Features) testingData.Dataview

