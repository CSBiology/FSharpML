(*** hide ***)
// This block of code is omitted in the generated HTML documentation. Use 
// it to define helpers that you do not want to show in the documentation.
//#r "../../packages/formatting/FSharp.Plotly/lib/netstandard2.0/Fsharp.Plotly.dll"
//#r "netstandard"
//open FSharp.Plotly
(**
Sample Spam detection
=========================

**)

#load "../../bin/FSharpML/netstandard2.0/FSharpML.fsx"
//#load "../../FSharpML.fsx"

open System;
open Microsoft.ML
open Microsoft.ML.Data;
open FSharpML
open FSharpML.EstimatorModel




let trainDataPath  = (__SOURCE_DIRECTORY__  + "./data/SMSSpamCollection.txt")

let mlContext = MLContext(seed = Nullable 1)

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
    |> EstimatorModel.appendBy (conversionValueMap ["ham"; "spam"] [false; true] [| struct ("Label", "LabelText") |])
    |> EstimatorModel.appendBy (fun mlc -> mlc.Transforms.Text.FeaturizeText("Features", "Message"))
    |> EstimatorModel.appendCacheCheckpoint
    |> EstimatorModel.appendBy (fun mlc -> mlc.BinaryClassification.Trainers.LogisticRegression("Label", "Features"))




// Evaluate the model using cross-validation.
// Cross-validation splits our dataset into 'folds', trains a model on some folds and 
// evaluates it on the remaining fold. We are using 5 folds so we get back 5 sets of scores.
// Let's compute the average AUC, which should be between 0.5 and 1 (higher is better).
let cvResults = mlContext.BinaryClassification.CrossValidate(data, estimatorModel.EstimatorChain |> Estimator.downcastEstimator, numFolds = 5,stratificationColumn = null)
let avgAuc = cvResults |> Seq.map (fun struct (metrics,_,_) -> metrics.Auc) |> Seq.average
//printfn "The AUC is %f" avgAuc


// Now let's train a model on the full dataset to help us get better results
let model = EstimatorModel.fit data estimatorModel
let out = TransformerModel.transform data model


model
|> Evaluation.BinaryClassificationEvalute() data

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


