namespace FSharpML

open System
open Microsoft.ML
open Microsoft.ML.Core.Data
open Microsoft.Data.DataView
open Microsoft.ML.Data
open System.Runtime.InteropServices



module DataModel =
    
    type TrainTestSplit = {
        TrainingData        : IDataView
        TestData            : IDataView
        SplitFraction       : float
        StratificationColumn: string option
        }
 
    type BinaryClassification =
        
        static member initTrainTestSplit
            (
                context:MLContext, 
                ?Testfraction:float,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let testFraction    = defaultArg Testfraction 0.
                let stratification  = defaultArg Stratification null
                let seed            = Option.toNullable Seed
                
                fun (data:IDataView)  ->                                                           
                    let train,test = (context.BinaryClassification.TrainTestSplit(data,testFraction,stratification,seed)).ToTuple() 
                    {TrainingData = train; TestData = test; SplitFraction = testFraction; StratificationColumn= Stratification}

    type MulticlassClassification =

        static member initTrainTestSplit
            (
                context:MLContext, 
                ?Testfraction:float,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let testFraction    = defaultArg Testfraction 0.
                let stratification  = defaultArg Stratification null
                let seed            = Option.toNullable Seed
                
                fun (data:IDataView)  ->                                                           
                    let train,test = (context.MulticlassClassification.TrainTestSplit(data,testFraction,stratification,seed)).ToTuple() 
                    {TrainingData = train; TestData = test; SplitFraction = testFraction; StratificationColumn= Stratification}
                    
    type Regression =
        
        static member initTrainTestSplit
            (
                context:MLContext, 
                ?Testfraction:float,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let testFraction    = defaultArg Testfraction 0.
                let stratification  = defaultArg Stratification null
                let seed            = Option.toNullable Seed
                
                fun (data:IDataView)  ->                                                           
                    let train,test = (context.Regression.TrainTestSplit(data,testFraction,stratification,seed)).ToTuple() 
                    {TrainingData = train; TestData = test; SplitFraction = testFraction; StratificationColumn= Stratification}
                    
    type Clustering =
 
        static member initTrainTestSplit
            (
                context:MLContext, 
                ?Testfraction:float,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let testFraction    = defaultArg Testfraction 0.
                let stratification  = defaultArg Stratification null
                let seed            = Option.toNullable Seed
                
                fun (data:IDataView)  ->                                                           
                    let train,test = (context.Clustering.TrainTestSplit(data,testFraction,stratification,seed)).ToTuple() 
                    {TrainingData = train; TestData = test; SplitFraction = testFraction; StratificationColumn= Stratification}
                    
                       
    type Ranking =
  
        static member initTrainTestSplit
            (
                context:MLContext, 
                ?Testfraction:float,
                ?Stratification : string,
                ?Seed : uint32 
            ) =
                let testFraction    = defaultArg Testfraction 0.
                let stratification  = defaultArg Stratification null
                let seed            = Option.toNullable Seed
                
                fun (data:IDataView)  ->                                                           
                    let train,test = (context.Ranking.TrainTestSplit(data,testFraction,stratification,seed)).ToTuple() 
                    {TrainingData = train; TestData = test; SplitFraction = testFraction; StratificationColumn= Stratification}
          

        
        //Not implemented in ML.net: AnomalyDetection