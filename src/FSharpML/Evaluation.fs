namespace FSharpML

open Microsoft.ML
open Microsoft.ML.Core.Data
open Microsoft.Data.DataView
open Microsoft.ML.Data


module Evaluation =
    
    type BinaryClassification =
        
        static member InitEvaluate
            (
                ?LabelCol : string,
                ?ScoreCol : string,
                ?ProbabilityCol : string,
                ?PredictedLabelCol : string
            ) =
                let label          = defaultArg LabelCol DefaultColumnNames.Label
                let score          = defaultArg ScoreCol DefaultColumnNames.Score
                let probability    = defaultArg ProbabilityCol DefaultColumnNames.Probability
                let predictedLabel = defaultArg PredictedLabelCol DefaultColumnNames.PredictedLabel
                
                fun (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>) ->                                       
                    let prediction = transformerModel |> TransformerModel.transform data
                    transformerModel.Context.BinaryClassification.Evaluate(prediction,label,score,probability,predictedLabel)

        static member InitEvaluateUncalibrated
            (
                ?LabelCol : string,
                ?ScoreCol : string,
                ?PredictedLabelCol : string
            ) =
                let label          = defaultArg LabelCol DefaultColumnNames.Label
                let score          = defaultArg ScoreCol DefaultColumnNames.Score
                let predictedLabel = defaultArg PredictedLabelCol DefaultColumnNames.PredictedLabel
                
                fun (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>) ->                                       
                    let prediction = transformerModel |> TransformerModel.transform data
                    transformerModel.Context.BinaryClassification.EvaluateNonCalibrated(prediction,label,score,predictedLabel)
    
    type Regression =
        
        static member InitEvaluate
            (
                ?LabelCol : string,
                ?ScoreCol : string
            ) =
                let label          = defaultArg LabelCol DefaultColumnNames.Label
                let score          = defaultArg ScoreCol DefaultColumnNames.Score
                
                fun (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>) ->                                       
                    let prediction = transformerModel |> TransformerModel.transform data
                    transformerModel.Context.Regression.Evaluate(prediction,label,score)

        static member InitEvaluateUncalibrated
            (
                ?LabelCol : string,
                ?ScoreCol : string,
                ?PredictedLabelCol : string
            ) =
                let label          = defaultArg LabelCol DefaultColumnNames.Label
                let score          = defaultArg ScoreCol DefaultColumnNames.Score
                let predictedLabel = defaultArg PredictedLabelCol DefaultColumnNames.PredictedLabel
                
                fun (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>) ->                                       
                    let prediction = transformerModel |> TransformerModel.transform data
                    transformerModel.Context.BinaryClassification.EvaluateNonCalibrated(prediction,label,score,predictedLabel)
                       
                       
    type Clustering =
        
        static member InitEvaluate
            (
                ?LabelCol : string,
                ?ScoreCol : string,
                ?FeaturesCol : string
            ) =
                let label          = defaultArg LabelCol DefaultColumnNames.Label
                let score          = defaultArg ScoreCol DefaultColumnNames.Score
                let features       = defaultArg FeaturesCol DefaultColumnNames.Features                
                
                fun (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>) ->                                       
                    let prediction = transformerModel |> TransformerModel.transform data
                    transformerModel.Context.Clustering.Evaluate(prediction, label, score, features)
            



// module TransformerModel =
//     open Microsoft.Data.DataView
//     open Microsoft.ML.Data

//     ///
//     let trainFrom (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) = 
//         let transformer = estimatorModel.EstimatorChain.Fit data
//         {TransformerModel.TransformerChain=transformer;TransformerModel.Context=estimatorModel.Context}    

