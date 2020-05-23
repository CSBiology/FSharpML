// --------------------------------------------------------------------------------------
// FAKE build script
// --------------------------------------------------------------------------------------

#r "paket:
nuget BlackFox.Fake.BuildTask
nuget Fake.Core.Target
nuget Fake.Core.Process
nuget Fake.Core.ReleaseNotes
nuget Fake.IO.FileSystem
nuget Fake.DotNet.Cli
nuget Fake.DotNet.MSBuild
nuget Fake.DotNet.AssemblyInfoFile
nuget Fake.DotNet.Paket
nuget Fake.DotNet.FSFormatting
nuget Fake.DotNet.Fsi
nuget Fake.DotNet.NuGet
nuget Fake.Api.Github
nuget Fake.DotNet.Testing.Expecto //"

#load ".fake/build.fsx/intellisense.fsx"

open BlackFox.Fake
open System.IO
open Fake.Core
open Fake.Core.TargetOperators
open Fake
open Fake.DotNet
open Fake.IO
open Fake.IO.FileSystemOperators
open Fake.IO.Globbing
open Fake.IO.Globbing.Operators
open Fake.DotNet.Testing
open Fake.Tools
open Fake.Api
open Fake.Tools.Git

Target.initEnvironment ()

[<AutoOpen>]
module MessagePrompts =

    let prompt (msg:string) =
        System.Console.Write(msg)
        System.Console.ReadLine().Trim()
        |> function | "" -> None | s -> Some s
        |> Option.map (fun s -> s.Replace ("\"","\\\""))

    let rec promptYesNo msg =
        match prompt (sprintf "%s [Yn]: " msg) with
        | Some "Y" | Some "y" -> true
        | Some "N" | Some "n" -> false
        | _ -> System.Console.WriteLine("Sorry, invalid answer"); promptYesNo msg

    let releaseMsg = """This will stage all uncommitted changes, push them to the origin and bump the release version to the latest number in the RELEASE_NOTES.md file. 
        Do you want to continue?"""

    let releaseDocsMsg = """This will push the docs to gh-pages. Remember building the docs prior to this. Do you want to continue?"""

// --------------------------------------------------------------------------------------
// START TODO: Provide project-specific details below
// --------------------------------------------------------------------------------------
let project         = "FSharpML"
let summary         = "Library for the FSharp friendly usage of the ML.NET project. For documentation visit: <https://csbiology.github.io/FSharpML/>"
let solutionFile    = "FSharpML.sln"
let configuration   = "Release"

//let testAssemblies = "tests/**/bin" </> configuration </> "**" </> "*Tests.exe"

let gitOwner = "CSBiology"
let gitHome = sprintf "%s/%s" "https://github.com" gitOwner
let gitName = "FSharpML"

let website = "/FSharpML"
let pkgDir = "pkg"

//Build configurations
let dotnetCoreConfiguration = DotNet.Custom configuration

// Read additional information from the release notes document
let release = ReleaseNotes.load "RELEASE_NOTES.md"

//---------------------------------------------------------------------------------------------------------------------------------
// Projects for different buildConfigs

// when building everything on windows (netstandard 2.0 + netfx)
let allProjectPaths =
    !! "src/**/*.fsproj"
    |>  Seq.map 
        (fun f -> (Path.getDirectory f))


//---------------------------------------------------------------------------------------------------------------------------------
//======================================================= Build Tasks =============================================================
//---------------------------------------------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------------------------------------------
// Clean build results

let clean = 
    BuildTask.create "clean" [] {
        Shell.cleanDirs [
            "bin"; "temp"; "pkg" 
            yield! allProjectPaths |> Seq.map (fun x -> x </> "bin")
            ]
    }

let cleanDocs = 
    BuildTask.create "cleanDocs" [] {
        Shell.cleanDirs ["docs"]
    }

// --------------------------------------------------------------------------------------------------------------------------------
// Generate assembly info files with the right version & up-to-date information. Clean first.

let assemblyInfo = 
    BuildTask.create "assemblyInfo" [clean.IfNeeded] {
        let getAssemblyInfoAttributes projectName =
            [ AssemblyInfo.Title (projectName)
              AssemblyInfo.Product project
              AssemblyInfo.Description summary
              AssemblyInfo.Version release.AssemblyVersion
              AssemblyInfo.FileVersion release.AssemblyVersion
              AssemblyInfo.Configuration configuration ]

        let getProjectDetails projectPath =
            let projectName = Path.GetFileNameWithoutExtension(projectPath)
            ( projectPath,
              projectName,
              Path.GetDirectoryName(projectPath),
              (getAssemblyInfoAttributes projectName)
            )

        !! "src/**/*.fsproj"
        |> Seq.map getProjectDetails
        |> Seq.iter 
            (fun (projFileName, _, folderName, attributes) ->
                AssemblyInfoFile.createFSharp (folderName </> "AssemblyInfo.fs") attributes
            )
    }


// --------------------------------------------------------------------------------------------------------------------------------
// Build library & test project. build assembly info first

let buildAll = 
    BuildTask.create "buildAll" [clean.IfNeeded; assemblyInfo] {
        solutionFile 
        |> DotNet.build (fun p -> 
            { p with
                Configuration = dotnetCoreConfiguration }
            )
    }


// --------------------------------------------------------------------------------------------------------------------------------
// Copies binaries from default VS location to expected bin folder
// But keeps a subdirectory structure for each project in the
// src folder to support multiple project outputs
// Build first.

let copyBinaries = 
        BuildTask.create "copyBinaries" [clean.IfNeeded; assemblyInfo.IfNeeded; buildAll] {
        !! "src/**/*.fsproj"
        |>  Seq.map 
            (fun f -> (Path.getDirectory f) </> "bin" </> configuration, "bin" </> (Path.GetFileNameWithoutExtension f))
        |>  Seq.iter 
            (fun (fromDir, toDir) -> 
                printfn "copy from %s to %s" fromDir toDir
                Shell.copyDir toDir fromDir (fun _ -> true)
            )
    }

// --------------------------------------------------------------------------------------
// Build a NuGet package. Build and test packages first

let buildReleasePackages = 
    BuildTask.create "BuildReleasePackages" [buildAll;] {
        Paket.pack(fun p ->
            { p with
                ToolType = ToolType.CreateLocalTool()
                OutputPath = pkgDir
                Version = release.NugetVersion
                ReleaseNotes = release.Notes |> String.toLines })
    }

let publishNugetPackages = 
    BuildTask.create "publishNugetPackages" [buildAll; buildReleasePackages] {
        Paket.push(fun p ->
            { p with
                WorkingDir = pkgDir
                ToolType = ToolType.CreateLocalTool()
                ApiKey = Environment.environVarOrDefault "NuGet-key" "" })
    }

// --------------------------------------------------------------------------------------
// Generate the documentation

let generateDocumentation = 
    BuildTask.create "generateDocumentation" [buildAll] {
        let result =
            DotNet.exec
                (fun p -> { p with WorkingDirectory = __SOURCE_DIRECTORY__ @@ "docsrc" @@ "tools" })
                "fsi"
                "--define:RELEASE --define:REFERENCE --define:HELP --exec generate.fsx"

        if not result.OK then 
            failwith "error generating docs" 
    }


// --------------------------------------------------------------------------------------
// Release Scripts

//#load "paket-files/fsharp/FAKE/modules/Octokit/Octokit.fsx"
//open Octokit
let askForDocReleaseConfirmation =
    BuildTask.create "askForDocReleaseConfirmation" [] {
        match promptYesNo releaseDocsMsg with | true -> () |_ -> failwith "Release canceled"
    }
let releaseDocsToGhPages = 
    BuildTask.create "releaseDocsToGhPages" [askForDocReleaseConfirmation;generateDocumentation] {
        let tempDocsDir = "temp/gh-pages"
        Shell.cleanDir tempDocsDir |> ignore
        Git.Repository.cloneSingleBranch "" (gitHome + "/" + gitName + ".git") "gh-pages" tempDocsDir
        Shell.copyRecursive "docs" tempDocsDir true |> printfn "%A"
        Git.Staging.stageAll tempDocsDir
        Git.Commit.exec tempDocsDir (sprintf "Update generated documentation for version %s" release.NugetVersion)
        Git.Branches.push tempDocsDir
    }

let buildLocalDocs = 
    BuildTask.create "buildLocalDocs" [generateDocumentation] {
        let tempDocsDir = "temp/localDocs"
        Shell.cleanDir tempDocsDir |> ignore
        Shell.copyRecursive "docs" tempDocsDir true  |> printfn "%A"
        Shell.replaceInFiles 
            (seq {
                yield "href=\"/" + project + "/","href=\""
                yield "src=\"/" + project + "/","src=\""}) 
            (Directory.EnumerateFiles tempDocsDir |> Seq.filter (fun x -> x.EndsWith(".html")))
    }


let askForReleaseConfirmation = 
    BuildTask.create "askForReleaseConfirmation" [] {
        match promptYesNo releaseMsg with | true -> () |_ -> failwith "Release canceled"
    }

let releaseOnGithub = 
    BuildTask.create "releaseOnGithub" [askForReleaseConfirmation; buildAll;] {
        Git.Staging.stageAll ""
        Git.Commit.exec "" (sprintf "Bump version to %s" release.NugetVersion)
        Git.Branches.push ""

        Git.Branches.tag "" release.NugetVersion
        Git.Branches.pushTag "" "upstream" release.NugetVersion
    }


let dotnetBuildChainLocal = 
    BuildTask.createEmpty "dotnetBuildChainLocal" [
        clean
        assemblyInfo
        buildAll
        copyBinaries
        cleanDocs
        generateDocumentation
        buildReleasePackages
    ]


BuildTask.runOrDefaultWithArguments dotnetBuildChainLocal
