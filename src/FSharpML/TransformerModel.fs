namespace FSharpML.TransformerModel

open Microsoft.ML
open Microsoft.ML.Core.Data

[<AutoOpen>]
module TransformerModel =
    open Microsoft.ML.Data

    type TransformerModel<'a when 'a :> ITransformer and 'a :not struct> = {
        TransformerChain : TransformerChain<'a>
        Context: MLContext
        }

    /// Returns the MLContext from an TransformerModel 
    let getContext (transformerModel:TransformerModel<_>) = transformerModel.Context

    /// Returns the EstimatorChain from an TransformerModel
    let getTransformerChain (transformerModel:TransformerModel<_>) = transformerModel.TransformerChain


    ///
    let transform data (transformerModel:TransformerModel<_>) =
        transformerModel.TransformerChain.Transform data

    /// Creates a prediction engine function for one-time predictions
    let createPredictionEngine<'a, 'input,'predictionResult 
        when 'input :not struct 
            and 'predictionResult : (new: unit -> 'predictionResult) 
            and 'predictionResult :not struct  
            and 'a :> ITransformer and 'a :not struct>   (transformerModel:TransformerModel<'a>) = 
    
        let pe = transformerModel.TransformerChain.CreatePredictionEngine<'input,'predictionResult>(transformerModel.Context)
        fun item -> pe.Predict(item)
