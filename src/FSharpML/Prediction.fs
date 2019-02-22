namespace FSharpML

open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Core.Data
open Microsoft.Data.DataView
open TransformerModel

/// Module to build an estimator model that represents the estimator procedure  
module Prediction =

    module BinaryClassification = 
        
        type BinaryClassificationPrediction = {
            Score: float
            Probability : float
            Label : bool 
            }

        let predictDefaultCols (transformerModel: TransformerModel<'a>) (items:seq<'b>) = 
            let data = transformerModel.Context.Data.ReadFromEnumerable items
            let pred = transformerModel |> TransformerModel.transform  data 
            let scores :seq<float32> = 
                pred.GetColumn(transformerModel.Context.Data.GetEnvironment(),DefaultColumnNames.Score)
            let probability :seq<float32> = 
                pred.GetColumn(transformerModel.Context.Data.GetEnvironment(),DefaultColumnNames.Probability)
            let label :seq<bool> = 
                pred.GetColumn(transformerModel.Context.Data.GetEnvironment(),DefaultColumnNames.Label)
            Seq.map3 (fun s p l -> {Score= float s; Probability = float p; Label = l }) scores probability label
            |> Seq.zip items