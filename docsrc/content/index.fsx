(*** hide ***)
// This block of code is omitted in the generated HTML documentation. Use 
// it to define helpers that you do not want to show in the documentation.
#r "netstandard"
#r "../../lib/Formatting/FSharp.Plotly.dll"
#I "../../"
open FSharp.Plotly
//open FSharp.Plotly

(**
FSharpML: Explore ML.Net in F#
==============================

FSharpML is a lightweight API writen in F# on top of the powerful machine learning framework [ML.Net](http://dot.net/ml) library. It is designed to enable users to explore ML.Net in a scriptable manner and maintaining the functional style of F#.
The samples are ported from the official site [Samples for ML.NET](https://github.com/dotnet/machinelearning-samples).

After installing the package via Nuget we can load the delivered reference script and start using ML.Net in conjunction with FSharpML.
*)

#I "./packages/FSharpML/lib/netstandard2.0/"
#load "FSharpML.fsx"

open System
open Microsoft.ML
open Microsoft.ML.Data;
open FSharpML


(**
Start by creating a model context (MLContext) and a data reader with the loading configuration.
*)

//Create the MLContext to share across components for deterministic results
let mlContext = MLContext(seed = Nullable 1) //Seed set to any number so you
                                             //have a deterministic environment

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
    
//Split dataset in two parts: TrainingDataset (80%) and TestDataset (20%)
let struct(trainingDataView, testingDataView) = 
    mlContext.Clustering.TrainTestSplit(fullData, testFraction = 0.2)

(*** hide ***)
[<CLIMutable>] 
type IrisData = {
    Label : float32
    SepalLength : float32
    SepalWidth: float32
    PetalLength : float32
    PetalWidth : float32    
} 
(*** hide ***)
let plot1 = 
    mlContext.CreateEnumerable<IrisData>(fullData,false)
    |> Seq.groupBy (fun items -> items.Label)
    |> Seq.map (fun (k,values) -> 
        let x = values |> Seq.map (fun items -> items.SepalLength) 
        let y = values |> Seq.map (fun items -> items.SepalWidth) 
        Chart.Point(x,y,Name=sprintf "Label: %.0f" k)
        )
    |> Chart.Combine
(*** include-value: plot1 ***)


(**
After initializing an model context (MLContext) we can start to build our model by appending transformer functions. The EstimatorModel (Model) holds the context and the chain of estimators (EstimatorChain) and is than fitted to the training data in a training step. The resulting TransformerModel serves as a predictor.
*)


//STEP 2: Process data, create and train the model 
let model = 
    EstimatorModel.create mlContext
    // Process data transformations in pipeline
    |> EstimatorModel.appendBy (fun mlc -> 
                                    mlc.Transforms.Concatenate
                                                        (
                                                            DefaultColumnNames.Features , 
                                                            "SepalLength", 
                                                            "SepalWidth", 
                                                            "PetalLength", 
                                                            "PetalWidth"
                                                        ) )
    // Create the model
    |> EstimatorModel.appendBy (fun mlc -> 
                                    mlc.Clustering.Trainers.KMeans
                                            (
                                                featureColumn = DefaultColumnNames.Features, 
                                                clustersCount = 3
                                            ) )
    // Train the model
    |> EstimatorModel.fit trainingDataView

(**
The resulting TransformerModel serves as a predictor and can be tested by predicting our test data and evaluating the accuracy of the model.
*)

// STEP3: Run the prediciton on the test data
let predictions =
    model
    |> TransformerModel.transform testingDataView

// STEP4: Evaluate the accuracy of the model
let metrics = 
    model
    |> Evaluation.Clustering.InitEvaluate(Score=DefaultColumnNames.Score, Features=DefaultColumnNames.Features) testingDataView
    

(**
For more detailed examples continue to explore the FSharpML documentation.
*)


(**
Contributing and copyright
--------------------------

The project is hosted on [GitHub][gh] where you can [report issues][issues], fork 
the project and submit pull requests. If you're adding a new public API, please also 
consider adding [samples][content] that can be turned into a documentation. You might
also want to read the [library design notes][readme] to understand how it works.

The library is available under Public Domain license, which allows modification and 
redistribution for both commercial and non-commercial purposes. For more information see the 
[License file][license] in the GitHub repository. 

*)