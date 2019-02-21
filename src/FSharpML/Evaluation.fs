namespace FSharpML

open Microsoft.ML
open Microsoft.ML.Core.Data
open Microsoft.Data.DataView
open Microsoft.ML.Data


module Evaluation =
    
    type BinaryClassification =
        
        static member InitEvaluate
            (
                ?Label : string,
                ?Score : string,
                ?Probability : string,
                ?PredictedLabel : string
            ) =
                let label          = defaultArg Label DefaultColumnNames.Label
                let score          = defaultArg Score DefaultColumnNames.Score
                let probability    = defaultArg Probability DefaultColumnNames.Probability
                let predictedLabel = defaultArg PredictedLabel DefaultColumnNames.PredictedLabel
                
                fun (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>) ->                                       
                    let prediction = transformerModel |> TransformerModel.transform data
                    transformerModel.Context.BinaryClassification.Evaluate(prediction,label,score,probability,predictedLabel)

        static member InitEvaluateUncalibrated
            (
                ?Label : string,
                ?Score : string,
                ?PredictedLabel : string
            ) =
                let label          = defaultArg Label DefaultColumnNames.Label
                let score          = defaultArg Score DefaultColumnNames.Score
                let predictedLabel = defaultArg PredictedLabel DefaultColumnNames.PredictedLabel
                
                fun (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>) ->                                       
                    let prediction = transformerModel |> TransformerModel.transform data
                    transformerModel.Context.BinaryClassification.EvaluateNonCalibrated(prediction,label,score,predictedLabel)


    type MulticlassClassification =
        
        static member InitEvaluate
            (
                ?Label : string,
                ?Score : string,
                ?PredictedLabel : string,
                ?TopK : int
            ) =
                let label          = defaultArg Label DefaultColumnNames.Label
                let score          = defaultArg Score DefaultColumnNames.Score
                let predictedLabel = defaultArg PredictedLabel DefaultColumnNames.PredictedLabel
                let topK           = defaultArg TopK 1
                
                fun (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>) ->                                       
                    let prediction = transformerModel |> TransformerModel.transform data
                    transformerModel.Context.MulticlassClassification.Evaluate(prediction,label,score,predictedLabel,topK)

    type Regression =
        
        static member InitEvaluate
            (
                ?Label : string,
                ?Score : string
            ) =
                let label          = defaultArg Label DefaultColumnNames.Label
                let score          = defaultArg Score DefaultColumnNames.Score
                
                fun (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>) ->                                       
                    let prediction = transformerModel |> TransformerModel.transform data
                    transformerModel.Context.Regression.Evaluate(prediction,label,score)

                       
    type Clustering =
        
        static member InitEvaluate
            (
                ?Label : string,
                ?Score : string,
                ?Features : string
            ) =
                let label          = defaultArg Label DefaultColumnNames.Label
                let score          = defaultArg Score DefaultColumnNames.Score
                let features       = defaultArg Features DefaultColumnNames.Features                
                
                fun (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>) ->                                       
                    let prediction = transformerModel |> TransformerModel.transform data
                    transformerModel.Context.Clustering.Evaluate(prediction, label, score, features)

    type Ranking =
        
        static member InitEvaluate
            (
                ?Label : string,
                ?groupID : string,
                ?score : string
            ) =
                let label          = defaultArg Label DefaultColumnNames.Label
                let groupID        = defaultArg groupID DefaultColumnNames.Score
                let score          = defaultArg score DefaultColumnNames.Probability
                
                fun (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>) ->                                       
                    let prediction = transformerModel |> TransformerModel.transform data
                    transformerModel.Context.Ranking.Evaluate(prediction,label,groupID,score)
// module TransformerModel =
//     open Microsoft.Data.DataView
//     open Microsoft.ML.Data

//     ///
//     let trainFrom (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) = 
//         let transformer = estimatorModel.EstimatorChain.Fit data
//         {TransformerModel.TransformerChain=transformer;TransformerModel.Context=estimatorModel.Context}    

