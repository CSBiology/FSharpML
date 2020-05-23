namespace FSharpML.EstimatorModel

open FSharpML
open Microsoft.ML
open Microsoft.ML.Data

/// Function to cross validate models
module CrossValidation = 
    open Microsoft.ML.Data    



    /// Function to evaluate binary classification models
    type BinaryClassification =
        
        /// Performs model cross-validation with additional parameters defined by the user
        static member crossValidateWith
            (
                ?NumFolds : int,
                ?Label : string,
                ?Stratification : string,
                ?Seed : int 
            ) =
                let numFolds       = defaultArg NumFolds 5
                let label          = defaultArg Label DefaultColumnNames.Label
                let stratification = defaultArg Stratification null
                let seed           = Option.toNullable Seed
                
                fun (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) ->                                       
                    let estimator = estimatorModel |> EstimatorModel.getEstimatorChain |> Estimator.downcastEstimator
                    estimatorModel.Context.BinaryClassification.CrossValidate(data,estimator,numFolds,label,stratification,seed)

        /// Performs non calibrated model cross-validation with additional parameters defined by the user
        static member crossValidateNonCalibratedWith
            (
                ?NumFolds : int,
                ?Label : string,
                ?Stratification : string,
                ?Seed : int
            ) =
                let numFolds       = defaultArg NumFolds 5
                let label          = defaultArg Label DefaultColumnNames.Label
                let stratification = defaultArg Stratification null
                let seed           = Option.toNullable Seed
                
                fun (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) ->                                       
                    let estimator = estimatorModel |> EstimatorModel.getEstimatorChain |> Estimator.downcastEstimator
                    estimatorModel.Context.BinaryClassification.CrossValidateNonCalibrated(data,estimator,numFolds,label,stratification,seed)
        
        /// Function to evaluate binary classification models
        module BinaryClassification =
            
            /// Performs model cross-validation using default parameters
            let crossValidate (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>)  =                                       
                    BinaryClassification.crossValidateWith() data estimatorModel

            /// Performs non calibrated model cross-validation using default parameters
            let crossValidateCalibrated (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>)  =                                       
                    BinaryClassification.crossValidateNonCalibratedWith() data estimatorModel
    
    /// Function to evaluate multiclass classification models
    type MulticlassClassification =
        
        /// Performs model cross-validation with additional parameters defined by the user
        static member crossValidateWith
            (
                ?NumFolds : int,
                ?Label : string,
                ?Stratification : string,
                ?Seed : int 
            ) =
                let numFolds       = defaultArg NumFolds 5
                let label          = defaultArg Label DefaultColumnNames.Label
                let stratification = defaultArg Stratification null
                let seed           = Option.toNullable Seed
                
                fun (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) ->                                       
                    let estimator = estimatorModel |> EstimatorModel.getEstimatorChain |> Estimator.downcastEstimator
                    estimatorModel.Context.MulticlassClassification.CrossValidate(data,estimator,numFolds,label,stratification,seed)
        

        /// Function to evaluate multiclass classification models
        module MulticlassClassification =
            
            /// Performs model cross-validation using default parameters
            let crossValidate (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) =
                MulticlassClassification.crossValidateWith() data estimatorModel


          
    type Regression =
        
        /// Performs model cross-validation with additional parameters defined by the user
        static member crossValidateWith
            (
                ?NumFolds : int,
                ?Label : string,
                ?Stratification : string,
                ?Seed : int 
            ) =
                let numFolds       = defaultArg NumFolds 5
                let label          = defaultArg Label DefaultColumnNames.Label
                let stratification = defaultArg Stratification null
                let seed           = Option.toNullable Seed                                
                fun (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) ->                                       
                    let estimator = estimatorModel |> EstimatorModel.getEstimatorChain |> Estimator.downcastEstimator
                    estimatorModel.Context.Regression.CrossValidate(data,estimator,numFolds,label,stratification,seed)

        module Regression =

            
            /// Performs model cross-validation using default parameters
            let crossValidate (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) =
                Regression.crossValidateWith() data estimatorModel            

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
        //            estimatorModel.Context.Regression.TrainTestSplit(data,testFraction,stratification,seed),estimatorModel
                           
    type Clustering =
               
        /// Performs model cross-validation with additional parameters defined by the user
        static member crossValidateWith
            (
                ?NumFolds : int,
                ?Label : string,
                ?Features:string,
                ?Stratification : string,
                ?Seed : int 
            ) =
                let numFolds       = defaultArg NumFolds 5
                let label          = defaultArg Label DefaultColumnNames.Label
                let features       = defaultArg Label DefaultColumnNames.Features
                let stratification = defaultArg Stratification null
                let seed           = Option.toNullable Seed                                
                fun (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) ->                                       
                    let estimator = estimatorModel |> EstimatorModel.getEstimatorChain |> Estimator.downcastEstimator
                    estimatorModel.Context.Clustering.CrossValidate(data,estimator,numFolds,features,label,stratification,seed)

        module Clustering =
            
            /// Performs model cross-validation using default parameters
            let crossValidate (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) =
                Clustering.crossValidateWith() data estimatorModel   


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
        //            estimatorModel.Context.Clustering.TrainTestSplit(data,testFraction,stratification,seed),estimatorModel
                    
                       
    type Ranking =
        
        /// Performs model cross-validation with additional parameters defined by the user
        static member crossValidateWith () = raise (System.NotImplementedException("Not implemented in ML.NEZ"))
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
                      
        module Ranking =
            
            /// Performs model cross-validation using default parameters
            let crossValidate (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) =
                 raise (System.NotImplementedException("Not implemented in ML.NEZ"))
                

    
        
        //Not implemented in ML.net: AnomalyDetection

