namespace FSharpML.TransformerModel

open System
open Microsoft.ML
open Microsoft.ML.Core.Data
open Microsoft.Data.DataView
open Microsoft.ML.Data
open System.Runtime.InteropServices
open FSharpML
 


/// Function to evaluate models
module Evaluation =

    /// Function to evaluate binary classification models
    type BinaryClassification =
        
        /// Evalutes model with given column names defined by the user
        static member evaluateWith
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

        /// Evalutes model with given column names defined by the user (non calibrated)
        static member evaluateNonCalibratedWith
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
        
        /// Function to evaluate binary classification models
        module BinaryClassification =
            
            /// Evalutes model using default column names
            let evaluate (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>) =                                       
                    BinaryClassification.evaluateWith() data transformerModel

            /// Evalutes model using default column names (non calibrated)
            let evaluateUncalibrated (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>) =                                       
                    BinaryClassification.evaluateNonCalibratedWith() data transformerModel
    
    /// Function to evaluate multiclass classification models
    type MulticlassClassification =
        
        /// Evalutes model with given column names defined by the user
        static member evaluateWith
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
        

        /// Function to evaluate multiclass classification models
        module MulticlassClassification =

            /// Evalutes model using default default parameters
            let evaluate (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>)  =
                MulticlassClassification.evaluateWith() data transformerModel
          
    type Regression =
        
        /// Evalutes model with given column names defined by the user
        static member evaluateWith
            (
                ?Label : string,
                ?Score : string
            ) =
                let label          = defaultArg Label DefaultColumnNames.Label
                let score          = defaultArg Score DefaultColumnNames.Score
                
                fun (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>) ->                                       
                    let prediction = transformerModel |> TransformerModel.transform data
                    transformerModel.Context.Regression.Evaluate(prediction,label,score)

        module Regression =

            /// Evalutes model using default default parameters
            let evaluate (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>)  =
                Regression.evaluateWith() data transformerModel
                           
    type Clustering =
        
        /// Evalutes model with given column names defined by the user
        static member evaluateWith
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

        module Clustering =

            /// Evalutes model using default default parameters
            let evaluate (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>)  =
                Clustering.evaluateWith() data transformerModel
                    
                       
    type Ranking =
        
        static member evaluateWith
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

        module Ranking =

            /// Evalutes model using default default parameters
            let evaluate (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>)  =
                Ranking.evaluateWith() data transformerModel

        //TODO: add direct training on test and evaluation on train
        //static member initTrainTestSplit
        //    (
        //        ?Testfraction:float,
        //        ?Stratification : string,
        //        ?Seed : uint32 
        //    ) =
        //        let testFraction    = defaultArg Testfraction 0.
        //        let stratification  = defaultArg Stratification null
        //        let seed            = Option.toNullable Seed
                
        //        fun (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) ->                                       
        //            estimatorModel.Context.Ranking.TrainTestSplit(data,testFraction,stratification,seed),estimatorModel
          
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