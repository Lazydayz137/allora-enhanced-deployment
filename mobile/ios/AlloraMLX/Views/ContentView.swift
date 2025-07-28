//
//  ContentView.swift
//  AlloraMLX
//
//  Created by Allora Team on 2024-01-28.
//

import SwiftUI

struct ContentView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var mlxManager: MLXManager
    
    var body: some View {
        TabView(selection: $appState.selectedTab) {
            DashboardView()
                .tabItem {
                    Label("Dashboard", systemImage: "chart.xyaxis.line")
                }
                .tag(AppState.Tab.dashboard)
            
            WorkersView()
                .tabItem {
                    Label("Workers", systemImage: "server.rack")
                }
                .tag(AppState.Tab.workers)
            
            ModelsView()
                .tabItem {
                    Label("Models", systemImage: "brain")
                }
                .tag(AppState.Tab.models)
            
            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
                .tag(AppState.Tab.settings)
        }
        .accentColor(.blue)
        .onAppear {
            if appState.showOnboarding {
                print("Show onboarding screen.")
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .environmentObject(AppState())
            .environmentObject(MLXManager())
    }
}

