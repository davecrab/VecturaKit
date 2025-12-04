// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "VecturaKitLite",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .tvOS(.v17),
        .visionOS(.v1),
        .watchOS(.v10),
    ],
    products: [
        .library(
            name: "VecturaExternalKit",
            targets: ["VecturaExternalKit"]
        ),
        .library(
            name: "VecturaCore",
            targets: ["VecturaCore"]
        ),
        .executable(
            name: "vectura-cli",
            targets: ["VecturaExternalCLI"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.4.0")
    ],
    targets: [
        .target(
            name: "VecturaCore",
            dependencies: [],
            cSettings: [
                .define("ACCELERATE_NEW_LAPACK"),
                .define("ACCELERATE_LAPACK_ILP64"),
            ]
        ),
        .target(
            name: "VecturaExternalKit",
            dependencies: ["VecturaCore"]
        ),
        .executableTarget(
            name: "VecturaExternalCLI",
            dependencies: [
                "VecturaExternalKit",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .testTarget(
            name: "VecturaExternalKitTests",
            dependencies: ["VecturaExternalKit"]
        ),
    ]
)
