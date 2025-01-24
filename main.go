package main

import (
	"bufio"
	"encoding/csv"
	"encoding/gob"
	"fmt"
	"log"
	"math"
	"os"
	"strings"
)

type Model struct {
	Questions []string
	Answers   []string
}

// ----------- Dataset Handling -----------

// LoadDataset reads a CSV file and returns the questions and answers.
func LoadDataset(filePath string) ([]string, []string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read file: %v", err)
	}

	var questions []string
	var answers []string
	for _, row := range records {
		if len(row) >= 2 {
			questions = append(questions, strings.ToLower(row[0]))
			answers = append(answers, strings.ToLower(row[1]))
		}
	}

	return questions, answers, nil
}

// ----------- Model Save & Load -----------

// SaveModel saves the trained model to a file.
func SaveModel(model Model, filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(model); err != nil {
		return fmt.Errorf("failed to encode model: %v", err)
	}

	return nil
}

// LoadModel loads a trained model from a file.
func LoadModel(filePath string) (Model, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return Model{}, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	var model Model
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&model); err != nil {
		return Model{}, fmt.Errorf("failed to decode model: %v", err)
	}

	return model, nil
}

// ----------- Chatbot Logic -----------

// Tokenize: Chuyển câu thành các từ (tách từ)
func Tokenize(text string) []string {
	return strings.Fields(strings.ToLower(text))
}

// Vectorize: Chuyển câu thành vector tần suất từ (bag-of-words)
func Vectorize(text string) map[string]int {
	tokens := Tokenize(text)
	vector := make(map[string]int)
	for _, token := range tokens {
		vector[token]++
	}
	return vector
}

// CalculateCosineSimilarity: Tính độ tương tự cosine giữa 2 vector
func CalculateCosineSimilarity(vec1, vec2 map[string]int) float64 {
	var dotProduct, magnitudeA, magnitudeB float64

	for key, val := range vec1 {
		dotProduct += float64(val * vec2[key])
		magnitudeA += float64(val * val)
	}

	for _, val := range vec2 {
		magnitudeB += float64(val * val)
	}

	if magnitudeA == 0 || magnitudeB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(magnitudeA) * math.Sqrt(magnitudeB))
}

// FindAnswer: Tìm câu trả lời gần nhất
func FindAnswer(model Model, question string) string {
	questionVec := Vectorize(question)
	bestSimilarity := 0.0
	bestAnswer := "Sorry, I don't understand the question."

	// Duyệt qua từng câu hỏi trong model
	for i, q := range model.Questions {
		qVec := Vectorize(q)
		similarity := CalculateCosineSimilarity(questionVec, qVec)

		// Nếu tìm thấy độ tương tự cao hơn, cập nhật câu trả lời
		if similarity > bestSimilarity {
			bestSimilarity = similarity
			bestAnswer = model.Answers[i]
		}
	}

	// Nếu có độ tương tự đủ cao, trả về câu trả lời
	if bestSimilarity > 0.1 { // Bạn có thể điều chỉnh ngưỡng này (0.1) tùy vào độ chính xác mong muốn
		return bestAnswer
	}
	return "Sorry, I don't understand the question."
}

// ----------- Main Function -----------

func main() {
	const datasetPath = "dataset.csv"
	const modelPath = "model.bin"

	fmt.Println("Welcome to the NLP Chatbot!")
	fmt.Println("1. Train Model")
	fmt.Println("2. Test Chatbot")
	fmt.Print("Choose an option: ")

	var choice int
	fmt.Scanln(&choice)

	switch choice {
	case 1: // Train the model
		questions, answers, err := LoadDataset(datasetPath)
		if err != nil {
			log.Fatalf("Error loading dataset: %v", err)
		}

		model := Model{
			Questions: questions,
			Answers:   answers,
		}

		if err := SaveModel(model, modelPath); err != nil {
			log.Fatalf("Error saving model: %v", err)
		}

		fmt.Println("Model training complete and saved to", modelPath)

	case 2: // Test the chatbot
		model, err := LoadModel(modelPath)
		if err != nil {
			log.Fatalf("Error loading model: %v", err)
		}

		fmt.Println("Chatbot is ready! Type your question:")

		// Sử dụng bufio.Scanner để đọc toàn bộ câu hỏi
		scanner := bufio.NewScanner(os.Stdin)
		for {
			fmt.Print("> ")
			scanner.Scan()
			question := scanner.Text()
			answer := FindAnswer(model, question)
			fmt.Println("Bot:", answer)
		}

	default:
		fmt.Println("Invalid choice. Exiting.")
	}
}
