namespace FSharpML

open System
open Microsoft.ML
open Microsoft.Data.DataView




module DataModel =

    type DataModel<'info> = {
        Context  : MLContext
        Dataview : IDataView        
        Metadata : 'info option
        }
    
    /// Creates a data model record (use createDataModelWith to include metadata information)
    let createDataModel context dataview =
        {Context=context;Dataview=dataview;Metadata=None}

    /// Creates a data model record with metadata information
    let createDataModelWith context metaData dataview =
        {Context=context;Dataview=dataview;Metadata=Some metaData}
    

    /// Returns the MLcontext
    let getContext (dataModel:DataModel<_>) = dataModel.Context

    /// Returns the data view
    let getDataview (dataModel:DataModel<_>) = dataModel.Dataview

    /// Try to return meta data else None
    let tryGetMetadata (dataModel:DataModel<_>) = 
        dataModel.Metadata

    /// Metadata info for a traintestsplit    
    type TrainTestSplitInfo = {
        SplitFraction       : float
        StratificationColumn: string option
        }
    
    // Creates metadata info for a traintestsplit    
    let createTrainTestSplitInfo splitFraction stratificationColumn =
        {SplitFraction=splitFraction; StratificationColumn=stratificationColumn}
 
    
    type BinaryClassification =
        
        /// Splits a dataset into the train set and the test set according to the given fraction. Respects the StratificationColumn if provided.
        static member trainTestSplitWith
            (
                ?Testfraction:float,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let testfraction    = defaultArg Testfraction 0.
                let stratification  = defaultArg Stratification null
                let seed            = Option.toNullable Seed
                
                fun (dataModel:DataModel<_>)  ->                                                           
                    let train,test = (dataModel.Context.BinaryClassification.TrainTestSplit(dataModel.Dataview,testfraction,stratification,seed)).ToTuple() 
                    let trainInfo  = createTrainTestSplitInfo  (1. - testfraction) Stratification
                    let testInfo   = createTrainTestSplitInfo testfraction Stratification
                    (
                        createDataModelWith dataModel.Context trainInfo train,
                        createDataModelWith dataModel.Context testInfo test                    
                    )
    
    module BinaryClassification =
        
        /// Splits a dataset into the train set and the test set according to the given fraction.
        let trainTestSplit (dataModel:DataModel<_>) testfraction =
            BinaryClassification.trainTestSplitWith(Testfraction=testfraction) dataModel


    type MulticlassClassification =
        
        /// Splits a dataset into the train set and the test set according to the given fraction. Respects the StratificationColumn if provided.
        static member trainTestSplitWith
            (                
                ?Testfraction:float,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let testFraction    = defaultArg Testfraction 0.
                let stratification  = defaultArg Stratification null
                let seed            = Option.toNullable Seed
                
                fun (dataModel:DataModel<_>)  ->                                                           
                    let struct (train,test) = dataModel.Context.MulticlassClassification.TrainTestSplit(dataModel.Dataview,testFraction,stratification,seed)
                    let trainInfo  = createTrainTestSplitInfo testFraction Stratification
                    let testInfo   = createTrainTestSplitInfo (1. - testFraction) Stratification
                    (
                        createDataModelWith dataModel.Context trainInfo train,
                        createDataModelWith dataModel.Context testInfo test                    
                    )
    
    module MulticlassClassification =
        
        /// Splits a dataset into the train set and the test set according to the given fraction.
        let trainTestSplit (dataModel:DataModel<_>) testfraction =
            BinaryClassification.trainTestSplitWith(Testfraction=testfraction) dataModel
                    

    type Regression =
        
        /// Splits a dataset into the train set and the test set according to the given fraction. Respects the StratificationColumn if provided.
        static member trainTestSplitWith
            (                
                ?Testfraction:float,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let testFraction    = defaultArg Testfraction 0.
                let stratification  = defaultArg Stratification null
                let seed            = Option.toNullable Seed
                
                fun (dataModel:DataModel<_>)  ->                                                           
                    let struct (train,test) = dataModel.Context.Regression.TrainTestSplit(dataModel.Dataview,testFraction,stratification,seed)                    
                    let trainInfo  = createTrainTestSplitInfo testFraction Stratification
                    let testInfo   = createTrainTestSplitInfo (1. - testFraction) Stratification
                    (
                        createDataModelWith dataModel.Context trainInfo train,
                        createDataModelWith dataModel.Context testInfo test                    
                    )
    
    module Regression =
        
        /// Splits a dataset into the train set and the test set according to the given fraction.
        let trainTestSplit (dataModel:DataModel<_>) testfraction =
            Regression.trainTestSplitWith(Testfraction=testfraction) dataModel                    

    type Clustering =
    
        /// Splits a dataset into the train set and the test set according to the given fraction. Respects the StratificationColumn if provided.
        static member trainTestSplitWith
            (                
                ?Testfraction:float,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let testFraction    = defaultArg Testfraction 0.
                let stratification  = defaultArg Stratification null
                let seed            = Option.toNullable Seed
                
                fun (dataModel:DataModel<_>)  ->                                                           
                    let struct (train,test) = dataModel.Context.Clustering.TrainTestSplit(dataModel.Dataview,testFraction,stratification,seed)                    
                    let trainInfo  = createTrainTestSplitInfo testFraction Stratification
                    let testInfo   = createTrainTestSplitInfo (1. - testFraction) Stratification
                    (
                        createDataModelWith dataModel.Context trainInfo train,
                        createDataModelWith dataModel.Context testInfo test                    
                    )
                    
    
    module Clustering =
        
        /// Splits a dataset into the train set and the test set according to the given fraction.
        let trainTestSplit (dataModel:DataModel<_>) testfraction =
            Clustering.trainTestSplitWith(Testfraction=testfraction) dataModel   
                    
                       
    type Ranking =
        
        /// Splits a dataset into the train set and the test set according to the given fraction. Respects the StratificationColumn if provided.
        static member trainTestSplitWith
            (                
                ?Testfraction:float,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let testFraction    = defaultArg Testfraction 0.
                let stratification  = defaultArg Stratification null
                let seed            = Option.toNullable Seed
                
                fun (dataModel:DataModel<_>)  ->                                                           
                    let struct (train,test) = dataModel.Context.Ranking.TrainTestSplit(dataModel.Dataview,testFraction,stratification,seed)                    
                    let trainInfo  = createTrainTestSplitInfo testFraction Stratification
                    let testInfo   = createTrainTestSplitInfo (1. - testFraction) Stratification
                    (
                        createDataModelWith dataModel.Context trainInfo train,
                        createDataModelWith dataModel.Context testInfo test                    
                    )

     module Ranking =
        
        /// Splits a dataset into the train set and the test set according to the given fraction.
        let trainTestSplit (dataModel:DataModel<_>) testfraction =
            Ranking.trainTestSplitWith(Testfraction=testfraction) dataModel          

        
        //Not implemented in ML.net: AnomalyDetection