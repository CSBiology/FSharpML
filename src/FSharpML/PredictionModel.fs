namespace FSharpML

open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Core.Data
open Microsoft.Data.DataView
open FSharpML.TransformerModel

/// Module to build an estimator model that represents the estimator procedure  
module PredictionModel =

    let x = "not implemented yet"
    //type PredictionModel<'a ,'b when  'a :not struct and 'b :not struct and 'b : (new: unit -> 'b)> = {
    //    PredictionEngine : PredictionEngine<'a,'b>
    //    Context: MLContext
    //    }

    //type PredictionResult<'a,'b> = {
    //    Item  : 'a
    //    Result:'b
    //    }


    //let predictionModelOf<'a ,'b when  'a :not struct and 'b :not struct and 'b : (new: unit -> 'b)> (transformerModel:TransformerModel.TransformerModel<_>) = 
    //    //let transformerChain' = transformerModel.TransformerChain |> Seq.toArray |> Transformer.createTransformerChainOf //fun x -> TransformerChain<ITransformer>(x)
    //    //let predictor = transformerChain'.CreatePredictionEngine<'a,'b>(transformerModel.Context)
    //    //{PredictionEngine=predictor;Context=transformerModel.Context}
    //    //transformerChain'
    //    box transformerModel.TransformerChain

    //let predict (predictionModel:PredictionModel<'a,'b>) (v:'a) =
    //    predictionModel.PredictionEngine.Predict(v)

    //let predictZip (predictionModel:PredictionModel<'a,'b>) (v:'a) =
    //    {Item=v;Result=predictionModel.PredictionEngine.Predict(v)}
    ////// Create a PredictionFunction from our model 
    //let predictor = newModel.CreatePredictionEngine<SpamInput, SpamPrediction>(mlContext);
