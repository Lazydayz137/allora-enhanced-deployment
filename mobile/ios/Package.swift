// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "AlloraMLX",
    platforms: [
        .iOS(.v16),
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "AlloraMLX",
            targets: ["AlloraMLX"]
        ),
    ],
    dependencies: [
        // MLX Swift packages
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.10.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-examples", from: "0.2.0"),
        
        // Networking
        .package(url: "https://github.com/Alamofire/Alamofire.git", from: "5.8.0"),
        
        // SSH for remote management
        .package(url: "https://github.com/Frugghi/SwiftSH.git", from: "1.7.4"),
        
        // Charts for visualizations
        .package(url: "https://github.com/AppPear/ChartView", from: "1.5.5"),
        
        // Async utilities
        .package(url: "https://github.com/pointfreeco/swift-async-algorithms", from: "1.0.0"),
        
        // KeychainAccess for secure storage
        .package(url: "https://github.com/kishikawakatsumi/KeychainAccess.git", from: "4.2.2")
    ],
    targets: [
        .target(
            name: "AlloraMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
                .product(name: "LLMMLX", package: "mlx-swift-examples"),
                .product(name: "Alamofire", package: "Alamofire"),
                .product(name: "SwiftSH", package: "SwiftSH"),
                .product(name: "SwiftUICharts", package: "ChartView"),
                .product(name: "AsyncAlgorithms", package: "swift-async-algorithms"),
                .product(name: "KeychainAccess", package: "KeychainAccess")
            ]
        ),
        .testTarget(
            name: "AlloraMLXTests",
            dependencies: ["AlloraMLX"]
        ),
    ]
)
