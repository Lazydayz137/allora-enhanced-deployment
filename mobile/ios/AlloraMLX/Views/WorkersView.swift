//
//  WorkersView.swift
//  AlloraMLX
//
//  Created by Allora Team on 2024-01-28.
//

import SwiftUI

struct WorkersView: View {
    @EnvironmentObject var networkManager: NetworkManager
    
    var body: some View {
        NavigationView {
            List(networkManager.workers) { worker in
                WorkerRow(worker: worker)
            }
            .navigationTitle("Workers")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button(action: addWorker) {
                        Label("Add Worker", systemImage: "plus")
                    }
                }
            }
        }
    }
    
    func addWorker() {
        // Logic to add a new worker
        print("Add new worker.")
    }
}

struct WorkerRow: View {
    let worker: Worker
    
    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text(worker.name)
                    .font(.headline)
                Text(worker.status.rawValue)
                    .font(.subheadline)
                    .foregroundColor(.gray)
            }
            Spacer()
            if worker.isActive {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
            } else {
                Image(systemName: "xmark.circle.fill")
                    .foregroundColor(.red)
            }
        }
    }
}

struct WorkersView_Previews: PreviewProvider {
    static var previews: some View {
        WorkersView()
            .environmentObject(NetworkManager())
    }
}

// MARK: - Mock Models for Preview
struct Worker: Identifiable {
    let id = UUID()
    let name: String
    let status: Status
    let isActive: Bool
    
    enum Status: String {
        case idle = "Idle"
        case running = "Running"
        case stopped = "Stopped"
    }
}

class NetworkManager: ObservableObject {
    @Published var workers: [Worker] = [
        Worker(name: "Worker 1", status: .running, isActive: true),
        Worker(name: "Worker 2", status: .idle, isActive: false),
        Worker(name: "Worker 3", status: .stopped, isActive: false)
    ]
}
