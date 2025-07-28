//
//  AlloraMLXApp.swift
//  AlloraMLX
//
//  Created by Allora Team on 2024-01-28.
//

import SwiftUI
import MLX
import MLXNN

@main
struct AlloraMLXApp: App {
    @StateObject private var appState = AppState()
    @StateObject private var mlxManager = MLXManager()
    @StateObject private var networkManager = NetworkManager()
    
    init() {
        // Configure app appearance
        configureAppearance()
        
        // Initialize MLX
        MLX.GPU.set(device: .gpu)
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .environmentObject(mlxManager)
                .environmentObject(networkManager)
                .task {
                    await appState.initialize()
                }
        }
    }
    
    private func configureAppearance() {
        // Configure navigation bar appearance
        let appearance = UINavigationBarAppearance()
        appearance.configureWithOpaqueBackground()
        appearance.backgroundColor = UIColor.systemBackground
        appearance.titleTextAttributes = [.foregroundColor: UIColor.label]
        appearance.largeTitleTextAttributes = [.foregroundColor: UIColor.label]
        
        UINavigationBar.appearance().standardAppearance = appearance
        UINavigationBar.appearance().compactAppearance = appearance
        UINavigationBar.appearance().scrollEdgeAppearance = appearance
        
        // Configure tab bar appearance
        let tabBarAppearance = UITabBarAppearance()
        tabBarAppearance.configureWithOpaqueBackground()
        tabBarAppearance.backgroundColor = UIColor.systemBackground
        
        UITabBar.appearance().standardAppearance = tabBarAppearance
        UITabBar.appearance().scrollEdgeAppearance = tabBarAppearance
    }
}

// MARK: - App State
@MainActor
class AppState: ObservableObject {
    @Published var isInitialized = false
    @Published var currentUser: User?
    @Published var selectedTab: Tab = .dashboard
    @Published var showOnboarding = false
    
    enum Tab: String, CaseIterable {
        case dashboard = "Dashboard"
        case workers = "Workers"
        case models = "Models"
        case settings = "Settings"
        
        var icon: String {
            switch self {
            case .dashboard: return "chart.xyaxis.line"
            case .workers: return "server.rack"
            case .models: return "brain"
            case .settings: return "gear"
            }
        }
    }
    
    func initialize() async {
        // Check for first launch
        if !UserDefaults.standard.bool(forKey: "hasCompletedOnboarding") {
            showOnboarding = true
        }
        
        // Load user data
        await loadUserData()
        
        isInitialized = true
    }
    
    private func loadUserData() async {
        // Load user from keychain or create new
        currentUser = User(
            id: UUID().uuidString,
            name: "Allora User",
            walletAddress: nil
        )
    }
}

// MARK: - User Model
struct User: Identifiable, Codable {
    let id: String
    let name: String
    var walletAddress: String?
    var joinDate: Date = Date()
}
