namespace FSharpML

open Microsoft.ML
open Microsoft.ML.Core.Data
open Microsoft.Data.DataView
open Microsoft.ML.Data
open System.Runtime.InteropServices

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

        static member InitCrossValidation
            (
                ?NumFolds : int,
                ?Label : string,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let numFolds       = defaultArg NumFolds 5
                let label          = defaultArg Label DefaultColumnNames.Label
                let stratification = defaultArg Stratification null
                let seed           = Option.toNullable Seed
                
                fun (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) ->                                       
                    let estimator = estimatorModel |> EstimatorModel.getEstimatorChain |> Estimator.downcastEstimator
                    estimatorModel.Context.BinaryClassification.CrossValidate(data,estimator,numFolds,label,stratification,seed)

        static member InitCrossValidationCalibrated
            (
                ?NumFolds : int,
                ?Label : string,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let numFolds       = defaultArg NumFolds 5
                let label          = defaultArg Label DefaultColumnNames.Label
                let stratification = defaultArg Stratification null
                let seed           = Option.toNullable Seed
                
                fun (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) ->                                       
                    let estimator = estimatorModel |> EstimatorModel.getEstimatorChain |> Estimator.downcastEstimator
                    estimatorModel.Context.BinaryClassification.CrossValidateNonCalibrated(data,estimator,numFolds,label,stratification,seed)
        
        static member initTrainTestSplit
            (
                ?Testfraction:float,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let testFraction    = defaultArg Testfraction 0.
                let stratification  = defaultArg Stratification null
                let seed            = Option.toNullable Seed
                
                fun (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) ->                                       
                    estimatorModel.Context.BinaryClassification.TrainTestSplit(data,testFraction,stratification,seed),estimatorModel
                    
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

        static member InitCrossValidation
            (
                ?NumFolds : int,
                ?Label : string,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let numFolds       = defaultArg NumFolds 5
                let label          = defaultArg Label DefaultColumnNames.Label
                let stratification = defaultArg Stratification null
                let seed           = Option.toNullable Seed
                
                fun (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) ->                                       
                    let estimator = estimatorModel |> EstimatorModel.getEstimatorChain |> Estimator.downcastEstimator
                    estimatorModel.Context.MulticlassClassification.CrossValidate(data,estimator,numFolds,label,stratification,seed)

        static member initTrainTestSplit
            (
                ?Testfraction:float,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let testFraction    = defaultArg Testfraction 0.
                let stratification  = defaultArg Stratification null
                let seed            = Option.toNullable Seed
                
                fun (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) ->                                       
                    estimatorModel.Context.MulticlassClassification.TrainTestSplit(data,testFraction,stratification,seed),estimatorModel
          
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

        static member InitCrossValidation
            (
                ?NumFolds : int,
                ?Label : string,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let numFolds       = defaultArg NumFolds 5
                let label          = defaultArg Label DefaultColumnNames.Label
                let stratification = defaultArg Stratification null
                let seed           = Option.toNullable Seed                                
                fun (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) ->                                       
                    let estimator = estimatorModel |> EstimatorModel.getEstimatorChain |> Estimator.downcastEstimator
                    estimatorModel.Context.Regression.CrossValidate(data,estimator,numFolds,label,stratification,seed)

        static member initTrainTestSplit
            (
                ?Testfraction:float,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let testFraction    = defaultArg Testfraction 0.
                let stratification  = defaultArg Stratification null
                let seed            = Option.toNullable Seed
                
                fun (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) ->                                       
                    estimatorModel.Context.Regression.TrainTestSplit(data,testFraction,stratification,seed),estimatorModel
                           
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

        static member InitCrossValidation
            (
                ?NumFolds : int,
                ?Label : string,
                ?Features:string,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let numFolds       = defaultArg NumFolds 5
                let label          = defaultArg Label DefaultColumnNames.Label
                let features       = defaultArg Label DefaultColumnNames.Features
                let stratification = defaultArg Stratification null
                let seed           = Option.toNullable Seed                                
                fun (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) ->                                       
                    let estimator = estimatorModel |> EstimatorModel.getEstimatorChain |> Estimator.downcastEstimator
                    estimatorModel.Context.Clustering.CrossValidate(data,estimator,numFolds,features,label,stratification,seed)

        static member initTrainTestSplit
            (
                ?Testfraction:float,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let testFraction    = defaultArg Testfraction 0.
                let stratification  = defaultArg Stratification null
                let seed            = Option.toNullable Seed
                
                fun (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) ->                                       
                    estimatorModel.Context.Clustering.TrainTestSplit(data,testFraction,stratification,seed),estimatorModel
                    
                       
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

        static member initTrainTestSplit
            (
                ?Testfraction:float,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let testFraction    = defaultArg Testfraction 0.
                let stratification  = defaultArg Stratification null
                let seed            = Option.toNullable Seed
                
                fun (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) ->                                       
                    estimatorModel.Context.Ranking.TrainTestSplit(data,testFraction,stratification,seed),estimatorModel
          
        //Not implemented in ML.net
        //static member InitCrossValidation
        //    (
        //        ?NumFolds : int,
        //        ?Label : string,
        //        ?Features:string,
        //        ?Stratification : string,
        //        ?Seed : uint32 
        //    ) =
        //        let numFolds       = defaultArg NumFolds 5
        //        let label          = defaultArg Label DefaultColumnNames.Label
        //        let features       = defaultArg Label DefaultColumnNames.Features
        //        let stratification = defaultArg Stratification null
        //        let seed           = Option.toNullable Seed                                
        //        fun (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) ->                                       
        //            let estimator = estimatorModel |> EstimatorModel.getEstimatorChain |> Estimator.downcastEstimator
        //            estimatorModel.Context.Ranking.CrossValidate(data,estimator,numFolds,features,label,stratification,seed)
    
        
        //Not implemented in ML.net: AnomalyDetection