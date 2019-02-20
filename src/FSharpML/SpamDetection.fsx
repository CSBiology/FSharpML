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


let conversionValueMap  k v c  = 
    fun (mlContext:MLContext) -> mlContext.Transforms.Conversion.ValueMap(k,v,c) 

let trainDataPath  = (@"C:\Users\david\Source\Repos\netCoreRepos\FSharpML\docsrc\content\data\SMSSpamCollection.txt"(*__SOURCE_DIRECTORY__ + @"\..\..\docsrc\content\data\SMSSpamCollection.txt"*))

let mlContext = MLContext(seed = Nullable 1)

let read  path  = 
    fun (mlContext:MLContext) -> mlContext.Data.ReadFromTextFile(path)


let reader = 
    mlContext.Data.CreateTextLoader(
        columns = 
            [|
                TextLoader.Column("LabelText" , Nullable DataKind.Text, 0)
                TextLoader.Column("Message" , Nullable DataKind.Text, 1)
            |],
            hasHeader = false,
            separatorChar = '\t')
    
let data = reader.Read(trainDataPath)

// Create the estimator which converts the text label to a bool then featurizes the text, and add a linear trainer.
let estimatorModel = 
    EstimatorModel.create mlContext
    |> EstimatorModel.map (conversionValueMap ["ham"; "spam"] [false; true] [| struct ("Label", "LabelText") |])
    |> EstimatorModel.map (fun mlc -> mlc.Transforms.Text.FeaturizeText("Features", "Message"))
    |> EstimatorModel.appendCacheCheckpoint
    |> EstimatorModel.map (fun mlc -> mlc.BinaryClassification.Trainers.LogisticRegression("Label", "Features"))


// Evaluate the model using cross-validation.
// Cross-validation splits our dataset into 'folds', trains a model on some folds and 
// evaluates it on the remaining fold. We are using 5 folds so we get back 5 sets of scores.
// Let's compute the average AUC, which should be between 0.5 and 1 (higher is better).
let cvResults = mlContext.BinaryClassification.CrossValidate(data, estimatorModel.EstimatorChain |> downcastEstimator, numFolds = 5);
let avgAuc = cvResults |> Seq.map (fun struct (metrics,_,_) -> metrics.Auc) |> Seq.average
//printfn "The AUC is %f" avgAuc


// Now let's train a model on the full dataset to help us get better results
let model = EstimatorModel.fit data estimatorModel
let out = TransformerModel.transform data model

/// run till here, then continue, breaks 
let x = out.Preview().ColumnView
/// run till here, works
//x |> Seq.map (fun x -> x.Column.Name)

//let downcastOtherPipeline (pipeline : ITransformer) =
//    match pipeline with
//    | :? IPredictionTransformer<IPredictor> as p -> p
//    | _ -> failwith "The pipeline has to be an instance of IEstimator<ITransformer>."

//let permutationMetrics = 
//    mlContext.BinaryClassification.PermutationFeatureImportance(model.LastTransformer |> downcastOtherPipeline,out,"Label","Features",false,permutationCount=1)

// The dataset we have is skewed, as there are many more non-spam messages than spam messages.
// While our model is relatively good at detecting the difference, this skewness leads it to always
// say the message is not spam. We deal with this by lowering the threshold of the predictor. In reality,
// it is useful to look at the precision-recall curve to identify the best possible threshold.
//let newModel = 
//    let lastTransformer = 
//        BinaryPredictionTransformer<IPredictorProducing<float32>>(
//            mlContext, 
//            model.LastTransformer.Model, 
//            model.GetOutputSchema(data.Schema), 
//            model.LastTransformer.FeatureColumn, 
//            threshold = 0.15f, 
//            thresholdColumn = DefaultColumnNames.Probability);
//    let parts = model |> Seq.toArray
//    parts.[parts.Length - 1] <- lastTransformer :> _
//    TransformerChain<ITransformer>(parts)


//// Create a PredictionFunction from our model 
//let predictor = newModel.CreatePredictionEngine<SpamInput, SpamPrediction>(mlContext);

//// Test a few examples
//[
//    "That's a great idea. It should work."
//    "free medicine winner! congratulations"
//    "Yes we should meet over the weekend!"
//    "you win pills and free entry vouchers"
//] 
//|> List.iter (classify predictor)


