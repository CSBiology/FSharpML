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
                    let prediciton = transformerModel |> TransformerModel.transform data
                    transformerModel.Context.BinaryClassification.Evaluate(prediciton,label,score,probability,predictedLabel)
                
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
                    let prediciton = transformerModel |> TransformerModel.transform data
                    transformerModel.Context.Clustering.Evaluate(prediciton, label, score, features)
            



// module TransformerModel =
//     open Microsoft.Data.DataView
//     open Microsoft.ML.Data

//     ///
//     let trainFrom (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) = 
//         let transformer = estimatorModel.EstimatorChain.Fit data
//         {TransformerModel.TransformerChain=transformer;TransformerModel.Context=estimatorModel.Context}    

