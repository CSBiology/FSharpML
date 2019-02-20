#load "../../FSharpML.fsx"

open System;
open Microsoft.ML
open Microsoft.ML.Data;
open Microsoft.Data.DataView
open Microsoft.ML.Transforms
open Microsoft.ML.Transforms.Normalizers
open Microsoft.ML.Core
open Microsoft.ML.Core.Data
open Microsoft.ML.Internal
open Microsoft.ML.Model

open System
open System.IO
open Microsoft.ML
open Microsoft.ML.Data
open System.Net
open System.IO.Compression
open Microsoft.ML.Core.Data
open Microsoft.ML.Transforms.Conversions
open System.Numerics
open FSharpML
open FSharpML.EstimatorModel
module DataStructures =
    open Microsoft.ML.Data

/// Holds information about Iris flower to be classified.
[<CLIMutable>]
type IrisData = {
    [<LoadColumn(0)>]
    Label : float32

    [<LoadColumn(1)>]
    SepalLength : float32
        
    [<LoadColumn(2)>]
    SepalWidth : float32
        
    [<LoadColumn(3)>]
    PetalLength : float32
        
    [<LoadColumn(4)>]
    PetalWidth : float32
} 

/// Result of Iris classification. The array holds probability of the flower to be one of setosa, virginica or versicolor.
[<CLIMutable>]
type IrisPrediction = {
        Score : float32 []
    }    


module SampleIrisData =
    let Iris1 = { Label = 0.f; SepalLength = 5.1f; SepalWidth = 3.3f; PetalLength = 1.6f; PetalWidth= 0.2f }
    let Iris2 = { Label = 0.f; SepalLength = 6.4f; SepalWidth = 3.1f; PetalLength = 5.5f; PetalWidth = 2.2f }
    let Iris3 = { Label = 0.f; SepalLength = 4.4f; SepalWidth = 3.1f; PetalLength = 2.5f; PetalWidth = 1.2f }

//let appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs().[0])

//let baseDatasetsLocation = @"../../../../Data";
//let trainDataPath = sprintf @"%s/iris-train.txt" baseDatasetsLocation
//let testDataPath = sprintf @"%s/iris-test.txt" baseDatasetsLocation

//let baseModelsPath = @"../../../../MLModels"
//let modelPath = sprintf @"%s/IrisClassificationModel.zip" baseModelsPath


//let buildTrainEvaluateAndSaveModel (mlContext : MLContext) =
    
//    // STEP 1: Common data loading configuration
//    let trainingDataView = mlContext.Data.ReadFromTextFile<IrisData>(trainDataPath, hasHeader = true)
//    let testDataView = mlContext.Data.ReadFromTextFile<IrisData>(testDataPath, hasHeader = true)

//    // STEP 2: Common data process configuration with pipeline data transformations
//    let dataProcessPipeline = 
//        mlContext.Transforms.Concatenate("Features", "SepalLength",
//                                                     "SepalWidth",
//                                                     "PetalLength",
//                                                     "PetalWidth")
//                            .AppendCacheCheckpoint(mlContext)

//    // STEP 3: Set the training algorithm, then append the trainer to the pipeline  
//    let trainer = mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumn = "Label", featureColumn = "Features")
//    let trainingPipeline = dataProcessPipeline.Append(trainer)




//    // STEP 4: Train the model fitting to the DataSet

//    //Measure training time
//    let watch = System.Diagnostics.Stopwatch.StartNew()

//    printfn "=============== Training the model ==============="
//    let trainedModel = trainingPipeline.Fit(trainingDataView)

//    //Stop measuring time
//    watch.Stop()
//    let elapsedMs = float watch.ElapsedMilliseconds
//    printfn "***** Training time: %f seconds *****" (elapsedMs/1000.)


//    // STEP 5: Evaluate the model and show accuracy stats
//    printfn "===== Evaluating Model's accuracy with Test data ====="
//    let predictions = trainedModel.Transform(testDataView)
//    let metrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label", "Score")

//    Common.ConsoleHelper.printMultiClassClassificationMetrics (trainer.ToString()) metrics

//    // STEP 6: Save/persist the trained model to a .ZIP file
//    use fs = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write)
//    mlContext.Model.Save(trainedModel, fs);

//    printfn "The model is saved to %s" modelPath


//let testSomePredictions (mlContext : MLContext) =
//    //Test Classification Predictions with some hard-coded samples 
//    use stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read)
//    let trainedModel = mlContext.Model.Load(stream);

//    // Create prediction engine related to the loaded trained model
//    let predEngine = trainedModel.CreatePredictionEngine<IrisData, IrisPrediction>(mlContext)

//    //Score sample 1
//    let resultprediction1 = predEngine.Predict(DataStructures.SampleIrisData.Iris1)

//    printfn "Actual: setosa.     Predicted probability: setosa:      %.4f" resultprediction1.Score.[0]
//    printfn "                                           versicolor:  %.4f" resultprediction1.Score.[1]
//    printfn "                                           virginica:   %.4f" resultprediction1.Score.[2]
//    printfn ""

//    //Score sample 2
//    let resultprediction2 = predEngine.Predict(DataStructures.SampleIrisData.Iris2);
//    printfn "Actual: virginica.  Predicted probability: setosa:      %.4f" resultprediction2.Score.[0]
//    printfn "                                           versicolor:  %.4f" resultprediction2.Score.[1]
//    printfn "                                           virginica:   %.4f" resultprediction2.Score.[2]
//    printfn ""

//    //Score sample 3
//    let resultprediction2 = predEngine.Predict(DataStructures.SampleIrisData.Iris3);
//    printfn "Actual: versicolor. Predicted probability: setosa:      %.4f" resultprediction2.Score.[0]
//    printfn "                                           versicolor:  %.4f" resultprediction2.Score.[1]
//    printfn "                                           virginica:   %.4f" resultprediction2.Score.[2]
//    printfn ""


//[<EntryPoint>]
//let main argv =
    
//    // Create MLContext to be shared across the model creation workflow objects 
//    // Set a random seed for repeatable/deterministic results across multiple trainings.
//    let mlContext = MLContext(seed = Nullable 0)

//    //1.
//    buildTrainEvaluateAndSaveModel mlContext

//    //2.
//    testSomePredictions mlContext

//    printfn "=============== End of process, hit any key to finish ==============="
//    Console.ReadKey() |> ignore
