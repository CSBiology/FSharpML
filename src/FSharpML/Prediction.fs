namespace FSharpML

open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Core.Data
open Microsoft.Data.DataView
open FSharpML.TransformerModel

/// Module to build an estimator model that represents the estimator procedure  
module Prediction =

    module BinaryClassification = 
        
        type BinaryClassificationPrediction = {
            Score: float
            Probability : float
            PredictedLabel : bool 
            }

        let predictDefaultCols (transformerModel: TransformerModel<'a>) (items:seq<'b>) = 
            let data = transformerModel.Context.Data.ReadFromEnumerable items
            let pred = transformerModel |> TransformerModel.transform data 
            let scores :seq<float32> = 
                pred.GetColumn(transformerModel.Context,DefaultColumnNames.Score)
            let probability :seq<float32> = 
                pred.GetColumn(transformerModel.Context,DefaultColumnNames.Probability)
            let predictedLabel :seq<bool> = 
                pred.GetColumn(transformerModel.Context,DefaultColumnNames.PredictedLabel)
            Seq.map3 (fun s p l -> {Score= float s; Probability = float p; PredictedLabel = l }) scores probability predictedLabel
            |> Seq.zip items