//
//  ModelsView.swift
//  AlloraMLX
//
//  Created by Allora Team on 2024-01-28.
//

import SwiftUI

struct ModelsView: View {
    @EnvironmentObject var mlxManager: MLXManager
    @State private var selectedModel: MLModel?
    @State private var showModelDetail = false
    
    var body: some View {
        NavigationView {
            List {
                ForEach(ModelCategory.allCases, id: \.self) { category in
                    Section(header: Text(category.rawValue)) {
                        ForEach(mlxManager.models.filter { $0.category == category }) { model in
                            ModelRow(model: model)
                                .onTapGesture {
                                    selectedModel = model
                                    showModelDetail = true
                                }
                        }
                    }
                }
            }
            .navigationTitle("Models")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button(action: downloadModels) {
                        Label("Download", systemImage: "square.and.arrow.down")
                    }
                }
            }
            .sheet(isPresented: $showModelDetail) {
                if let model = selectedModel {
                    ModelDetailView(model: model)
                        .environmentObject(mlxManager)
                }
            }
        }
    }
    
    func downloadModels() {
        Task {
            await mlxManager.downloadAvailableModels()
        }
    }
}

struct ModelRow: View {
    let model: MLModel
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(model.name)
                    .font(.headline)
                Text(model.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
                HStack {
                    Label("\(model.sizeInMB) MB", systemImage: "internaldrive")
                    Spacer()
                    if model.isDownloaded {
                        Label("Ready", systemImage: "checkmark.circle.fill")
                            .foregroundColor(.green)
                    }
                }
                .font(.caption2)
            }
            .padding(.vertical, 4)
        }
    }
}

struct ModelDetailView: View {
    let model: MLModel
    @EnvironmentObject var mlxManager: MLXManager
    @State private var inputText = ""
    @State private var outputText = ""
    @State private var isProcessing = false
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Model Info
                    GroupBox {
                        VStack(alignment: .leading, spacing: 12) {
                            Text(model.name)
                                .font(.title2)
                                .bold()
                            
                            Text(model.description)
                                .font(.body)
                                .foregroundColor(.secondary)
                            
                            HStack {
                                Label("Size: \(model.sizeInMB) MB", systemImage: "internaldrive")
                                Spacer()
                                Label("v\(model.version)", systemImage: "number")
                            }
                            .font(.caption)
                            .foregroundColor(.secondary)
                        }
                    }
                    
                    // Input Section
                    GroupBox {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Input")
                                .font(.headline)
                            
                            if model.category == .textGeneration {
                                TextEditor(text: $inputText)
                                    .frame(height: 100)
                                    .padding(8)
                                    .background(Color(.systemGray6))
                                    .cornerRadius(8)
                            } else if model.category == .pricePrediction {
                                TextField("Enter parameters...", text: $inputText)
                                    .textFieldStyle(RoundedBorderTextFieldStyle())
                            }
                        }
                    }
                    
                    // Run Button
                    Button(action: runInference) {
                        HStack {
                            if isProcessing {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle())
                                    .scaleEffect(0.8)
                            } else {
                                Image(systemName: "play.fill")
                            }
                            Text(isProcessing ? "Processing..." : "Run Inference")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(model.isDownloaded ? Color.blue : Color.gray)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }
                    .disabled(!model.isDownloaded || isProcessing)
                    
                    // Output Section
                    if !outputText.isEmpty {
                        GroupBox {
                            VStack(alignment: .leading, spacing: 12) {
                                Text("Output")
                                    .font(.headline)
                                
                                Text(outputText)
                                    .font(.body)
                                    .padding(8)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .background(Color(.systemGray6))
                                    .cornerRadius(8)
                            }
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Model Details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") {
                        dismiss()
                    }
                }
            }
        }
    }
    
    func runInference() {
        Task {
            isProcessing = true
            outputText = await mlxManager.runInference(model: model, input: inputText)
            isProcessing = false
        }
    }
}

// MARK: - Model Types
enum ModelCategory: String, CaseIterable {
    case textGeneration = "Text Generation"
    case pricePrediction = "Price Prediction"
    case imageClassification = "Image Classification"
}

struct MLModel: Identifiable {
    let id = UUID()
    let name: String
    let description: String
    let category: ModelCategory
    let sizeInMB: Int
    let version: String
    let isDownloaded: Bool
    let modelPath: String?
}

struct ModelsView_Previews: PreviewProvider {
    static var previews: some View {
        ModelsView()
            .environmentObject(MLXManager())
    }
}
