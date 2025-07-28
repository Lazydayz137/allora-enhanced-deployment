//
//  SettingsView.swift
//  AlloraMLX
//
//  Created by Allora Team on 2024-01-28.
//

import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var appState: AppState
    @State private var enableNotifications = true
    @State private var saveActivity = true
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("General")) {
                    Toggle(isOn: $enableNotifications) {
                        Text("Enable Notifications")
                    }
                    Toggle(isOn: $saveActivity) {
                        Text("Save Activity Logs")
                    }
                }
                
                Section(header: Text("About")) {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text("1.0.0")
                            .foregroundColor(.gray)
                    }
                    NavigationLink(destination: AboutView()) {
                        Text("About AlloraMLX")
                    }
                }
                
                Section(header: Text("Support")) {
                    Button("Contact Support") {
                        // Contact support action
                    }
                }
            }
            .navigationTitle("Settings")
        }
    }
}

struct AboutView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("AlloraMLX")
                .font(.largeTitle)
                .bold()
            Text("Version 1.0.0")
                .foregroundColor(.secondary)
            Text("AlloraMLX is a powerful platform designed to streamline ML model deployment and management on Apple devices.")
                .font(.body)
            Spacer()
        }
        .padding()
        .navigationTitle("About AlloraMLX")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct SettingsView_Previews: PreviewProvider {
    static var previews: some View {
        SettingsView()
            .environmentObject(AppState())
    }
}
