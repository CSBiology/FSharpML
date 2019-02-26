(*** hide ***)
// This block of code is omitted in the generated HTML documentation. Use 
// it to define helpers that you do not want to show in the documentation.
#r "netstandard"
#r "../../lib/Formatting/FSharp.Plotly.dll"
open FSharp.Plotly
//open FSharp.Plotly
(**
Sample Spam detection
=========================

**)


(**
Introduction
============

FSharpML is a functional-friendly lightweight wrapper of the powerful [ML.Net](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet) library. It is designed to enable users to explore ML.Net in a scriptable manner, while maintaining the functional style of F#.

After installing the package via Nuget we can load the delivered reference script and start using ML.Net in conjunction with FSharpML.
*)


#load "../../FSharpML.fsx"



open System;
open Microsoft.ML
open Microsoft.ML.Data;
open FSharpML
open Microsoft.ML.Core.Data
open TransformerModel
open System.Data
(**
To get a feel how this library handles ML.Net operations we rebuild the [Spam Detection tutorial](https://github.com/dotnet/machinelearning-samples/tree/master/samples/fsharp/getting-started/BinaryClassification_SpamDetection) given by ML.Net. 
We will start by instantiating a MLContext, the heart of the ML.Net API and intended to serve as a method catalog. We will now use it to set a scope on data stored in a text file. The method name might be misleading, but ML.Net readers are lazy and
the reading process will start when the data is processed [(see).](https://github.com/dotnet/machinelearning/blob/master/docs/code/IDataViewTypeSystem.md#standard-column-types)
*)

let mlContext = MLContext(seed = Nullable 1)

let trainDataPath  = (__SOURCE_DIRECTORY__  + "./data/SMSSpamCollection.txt")
    
let data = 
    mlContext.Data.ReadFromTextFile( 
            path = trainDataPath,
            columns = 
                [|
                    TextLoader.Column("LabelText" , Nullable DataKind.Text, 0)
                    TextLoader.Column("Message" , Nullable DataKind.Text, 1)
                |],
            hasHeader = false,
            separatorChar = '\t')

(**
Now that we told our interactive environment about the data we can start thinking about a model (EstimatorChain in the ML.Net jargon)
we want to build. As the MLContext serves as a catalog we will use it to draw transformations that can be appended to form a estimator chain.
At this point we will see FSharpML comming into play enabling us to use the beloved pipelining style familiar to FSharp users.
We will now create an EstimatorChain which converts the text label to a bool then featurizes the text, and add a linear trainer.
*)

let estimatorChain = 
    EstimatorChain()
    |> Estimator.append (mlContext.Transforms.Conversion.ValueMap(["ham"; "spam"],[false; true],[| struct (DefaultColumnNames.Label, "LabelText") |]))
    |> Estimator.append (mlContext.Transforms.Text.FeaturizeText(DefaultColumnNames.Features, "Message"))
    |> (Estimator.appendCacheCheckpoint mlContext)
    |> Estimator.append (mlContext.BinaryClassification.Trainers.StochasticDualCoordinateAscent(DefaultColumnNames.Label, DefaultColumnNames.Features))

(**
This is already pretty fsharp-friendly but we thought we could even closer by releaving us from carring around our instance of the MLContext
explicitly. For this we created the type EstimatorModel which contains our EstimatorChain and the context. By Calling append by we only have to provide a
lambda expression were we can define which method we want of our context.
*)

let estimatorModel = 
    EstimatorModel.create mlContext
    |> EstimatorModel.appendBy (fun mlc -> mlc.Transforms.Conversion.ValueMap(["ham"; "spam"],[false; true],[| struct (DefaultColumnNames.Label, "LabelText") |]))
    |> EstimatorModel.appendBy (fun mlc -> mlc.Transforms.Text.FeaturizeText(DefaultColumnNames.Features, "Message"))
    |> EstimatorModel.appendCacheCheckpoint
    |> EstimatorModel.appendBy (fun mlc -> mlc.BinaryClassification.Trainers.StochasticDualCoordinateAscent(DefaultColumnNames.Label, DefaultColumnNames.Features))

(**
Way better. Now we can concentrate on machine learning. So lets start by fitting our EstimatorModel to the complete data set.
The return value of this process is a so called TransformerModel, which contains a trained EstimatorChain
and can be used to transform unseen data. For this we want to split the data we previously put in scope into 
two fractions. One to train the model and a remainder to evaluate the model. 
*)

let trainTestSplit = 
    data
    |> Data.BinaryClassification.initTrainTestSplit(estimatorModel.Context,Testfraction=0.1) 

let trainedModel = 
    estimatorModel
    |> EstimatorModel.fit trainTestSplit.TrainingData

let evaluationMetrics = 
    trainedModel
    |> Evaluation.BinaryClassification.evaluate trainTestSplit.TestData

let scoredData = 
    trainedModel
    |> TransformerModel.transform data 

evaluationMetrics.Accuracy

(**
Now that we can examine the metrics of our model evaluation, see that we have a accuracy of 0.99 
and be tempted to use it in production so lets test it first with some examples.
*)

type SpamInput = 
    {
        LabelText : string
        Message : string
    }

let exampleData = 
    [
        "That's a great idea. It should work."
        "free medicine winner! congratulations"
        "Yes we should meet over the weekend!"
        "you win pills and free entry vouchers"
    ] 
    |> List.map (fun message ->{LabelText = ""; Message = message})

exampleData 
|> (Prediction.BinaryClassification.predictDefaultCols trainedModel)
|> Array.ofSeq

(**
As we see, even so our accuracy when evaluating the model on the test data set was very high, it does not set the correct lable true, to the
second and the fourth message which look a lot like spam. Lets examine our training data set:
*)

let labels,counts =
    scoredData.GetColumn<bool>(mlContext,DefaultColumnNames.Label)
    |> Seq.countBy id
    |> Seq.unzip
    
let LabelDist = 
    Chart.Doughnut(counts,labels)
    
let probabilityHist =
    scoredData.GetColumn<bool>(mlContext,DefaultColumnNames.Label) 
    |> Seq.zip (scoredData.GetColumn<float32>(mlContext,DefaultColumnNames.Probability))
    |> Seq.groupBy snd 
    |> Seq.map (fun (label,x) ->
                    x
                    |> Seq.map fst
                    |> Chart.Histogram
                )
    |> Chart.Combine


let ScoreHist =
    scoredData.GetColumn<bool>(mlContext,DefaultColumnNames.Label) 
    |> Seq.zip (scoredData.GetColumn<float32>(mlContext,DefaultColumnNames.Score))
    |> Seq.groupBy snd 
    |> Seq.map (fun (label,x) ->
                    x
                    |> Seq.map fst
                    |> Chart.Histogram
                )
    |> Chart.Combine


let stackedChart = 
    [probabilityHist;ScoreHist]
    |> Chart.Stack(2)
    |> Chart.withSize(900.,600.)
    
(**
The chart clearly shows that the data we learned uppon is highly inhomogenous. We have a lot more ham than spam, which is generally preferable but 
but our models labeling threshold is clearly to high. Lets have a look at the precision recall curves of our model. For this we will evaluate the model
with different thresholds and plot both
*)
let idea = 
    trainedModel
    |> TransformerModel.transform data

let thresholdVSPrecicionAndRecall = 
    [-0.05 .. 0.05 .. 0.95]
    |> List.map (fun threshold ->
                    let newModel = 
                        let lastTransformer = 
                            BinaryPredictionTransformer<IPredictorProducing<float32>>(
                                trainedModel.Context, 
                                trainedModel.TransformerChain.LastTransformer.Model, 
                                trainedModel.TransformerChain.GetOutputSchema(idea.Schema), 
                                trainedModel.TransformerChain.LastTransformer.FeatureColumn, 
                                threshold = float32 threshold, 
                                thresholdColumn = DefaultColumnNames.Probability)
                        let parts = 
                            trainedModel.TransformerChain 
                            |> Seq.toArray
                            |> fun x ->x.[..x.Length-2] 
                        
                        printfn "%A, %A" lastTransformer.Threshold lastTransformer.ThresholdColumn
                        TransformerChain<ITransformer>(parts).Append(lastTransformer)
                    let newModel' = 
                        {TransformerModel.TransformerChain = newModel;Context=trainedModel.Context}
                        |> Evaluation.BinaryClassification.evaluate trainTestSplit.TestData
                    
                    threshold,newModel'.Accuracy
                    //threshold,
                    //exampleData 
                    //|> (Prediction.BinaryClassification.predictDefaultCols newModel')
                    //|> Array.ofSeq
                )




///
// Evaluate the model using cross-validation.
// Cross-validation splits our dataset into 'folds', trains a model on some folds and 
// evaluates it on the remaining fold. We are using 5 folds so we get back 5 sets of scores.
// Let's compute the average AUC, which should be between 0.5 and 1 (higher is better).
//let cvResults = mlContext.BinaryClassification.CrossValidate(data, estimatorModel.EstimatorChain |> Estimator.downcastEstimator, numFolds = 5,stratificationColumn = null)
//let avgAuc = cvResults |> Seq.map (fun struct (metrics,_,_) -> metrics.Auc) |> Seq.average
////printfn "The AUC is %f" avgAuc




//model
//|> Evaluation.BinaryClassificationEvalute() data

///// run till here, then continue, breaks 
//let x = out.Preview().ColumnView
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

