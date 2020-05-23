namespace FSharpML

open Microsoft.ML
open Microsoft.ML.Data


module FeatureImportance =

    let InitPermutationFeatureImportance() = System.MissingMethodException("Method is not implemented yet")
    //type BinaryClassification =
        
    //    static member InitPermutationFeatureImportance
    //        (
    //            ?Label : string,
    //            ?Features : string,
    //            ?UseFeatureWeightFilter:bool,
    //            ?TopExamples : int,
    //            ?PermutationCount: int
    //        ) =
    //            let label                   = defaultArg Label DefaultColumnNames.Label
    //            let features                = defaultArg Features DefaultColumnNames.Features
    //            let useFeatureWeightFilter  = defaultArg UseFeatureWeightFilter false
    //            let topExamples             = Option.toNullable TopExamples
    //            let permutationCount        = defaultArg PermutationCount 1
                
    //            fun (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>) ->                                       
    //                let prediction = transformerModel |> TransformerModel.transform data
    //                transformerModel.Context.BinaryClassification.PermutationFeatureImportance(transformerModel.TransformerChain.LastTransformer |> Transformer.downcastTransformer,prediction,label,features,useFeatureWeightFilter,topExamples,permutationCount)
        
    //type MulticlassClassification =
        
    //    static member InitPermutationFeatureImportance
    //        (
    //            ?Label : string,
    //            ?Features : string,
    //            ?UseFeatureWeightFilter:bool,
    //            ?TopExamples : int,
    //            ?PermutationCount: int
    //        ) =
    //            let label                   = defaultArg Label DefaultColumnNames.Label
    //            let features                = defaultArg Features DefaultColumnNames.Features
    //            let useFeatureWeightFilter  = defaultArg UseFeatureWeightFilter false
    //            let topExamples             = Option.toNullable TopExamples
    //            let permutationCount        = defaultArg PermutationCount 1
                
    //            fun (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>) ->                                       
    //                let prediction = transformerModel |> TransformerModel.transform data
    //                transformerModel.Context.MulticlassClassification.PermutationFeatureImportance(transformerModel.TransformerChain.LastTransformer |> Transformer.downcastTransformer,prediction,label,features,useFeatureWeightFilter,topExamples,permutationCount)
        
    //type Regression =
        
    //    static member InitPermutationFeatureImportance
    //        (
    //            ?Label : string,
    //            ?Features : string,
    //            ?UseFeatureWeightFilter:bool,
    //            ?TopExamples : int,
    //            ?PermutationCount: int
    //        ) =
    //            let label                   = defaultArg Label DefaultColumnNames.Label
    //            let features                = defaultArg Features DefaultColumnNames.Features
    //            let useFeatureWeightFilter  = defaultArg UseFeatureWeightFilter false
    //            let topExamples             = Option.toNullable TopExamples
    //            let permutationCount        = defaultArg PermutationCount 1
                
    //            fun (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>) ->                                       
    //                let prediction = transformerModel |> TransformerModel.transform data
    //                transformerModel.Context.Regression.PermutationFeatureImportance(transformerModel.TransformerChain.LastTransformer |> Transformer.downcastTransformer,prediction,label,features,useFeatureWeightFilter,topExamples,permutationCount)
        
    ///// Not implemented in ML.Net
    ////type Clustering =
        
    ////    static member InitPermutationFeatureImportance
    ////        (
    ////            ?Label : string,
    ////            ?Features : string,
    ////            ?UseFeatureWeightFilter:bool,
    ////            ?TopExamples : int
    ////        ) =
    ////            let label                   = defaultArg Label DefaultColumnNames.Label
    ////            let features                = defaultArg Features DefaultColumnNames.Features
    ////            let useFeatureWeightFilter  = defaultArg UseFeatureWeightFilter false
    ////            let topExamples             = Option.toNullable TopExamples
                
    ////            fun (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>) ->                                       
    ////                let prediction = transformerModel |> TransformerModel.transform data
    ////                transformerModel.Context.Clustering.PermutationFeatureImportance(transformerModel.TransformerChain.LastTransformer,prediction,label,features,useFeatureWeightFilter,topExamples)   
                       
    //type Ranking =
        
    //    static member InitPermutationFeatureImportance
    //        (
    //            ?Label : string,
    //            ?GroupID: string,
    //            ?Features : string,
    //            ?UseFeatureWeightFilter:bool,
    //            ?TopExamples : int
    //        ) =
    //            let label                   = defaultArg Label DefaultColumnNames.Label
    //            let groupID                 = defaultArg GroupID DefaultColumnNames.GroupId
    //            let features                = defaultArg Features DefaultColumnNames.Features
    //            let useFeatureWeightFilter  = defaultArg UseFeatureWeightFilter false
    //            let topExamples             = Option.toNullable TopExamples
                
    //            fun (data:IDataView) (transformerModel:TransformerModel.TransformerModel<_>) ->                                       
    //                let prediction = transformerModel |> TransformerModel.transform data
    //                transformerModel.Context.Ranking.PermutationFeatureImportance(transformerModel.TransformerChain.LastTransformer |> Transformer.downcastTransformer,prediction,label,groupID,features,useFeatureWeightFilter,topExamples)
        
          
    ////Not implemented in ML.net
    ////static member InitCrossValidation
    ////    (
    ////        ?NumFolds : int,
    ////        ?Label : string,
    ////        ?Features:string,
    ////        ?Stratification : string,
    ////        ?Seed : uint32 
    ////    ) =
    ////        let numFolds       = defaultArg NumFolds 5
    ////        let label          = defaultArg Label DefaultColumnNames.Label
    ////        let features       = defaultArg Label DefaultColumnNames.Features
    ////        let stratification = defaultArg Stratification null
    ////        let seed           = Option.toNullable Seed                                
    ////        fun (data:IDataView) (estimatorModel:EstimatorModel.EstimatorModel<_>) ->                                       
    ////            let estimator = estimatorModel |> EstimatorModel.getEstimatorChain |> Estimator.downcastEstimator
    ////            estimatorModel.Context.Ranking.CrossValidate(data,estimator,numFolds,features,label,stratification,seed)
        
    //    /// not implemented: AnomalyDetection