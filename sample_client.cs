// Sample C# client for room interior analysis
// Both snippets use the OpenAI .NET SDK: dotnet add package OpenAI
// Both accept: system prompt + image + user prompt → JSON response

using OpenAI;
using OpenAI.Chat;

// ============================================================================
// 1. OpenAI (GPT-4.1)
// ============================================================================

async Task<string> AnalyzeWithOpenAI(string imagePath, string systemPrompt, string userPrompt)
{
    var client = new ChatClient("gpt-4.1", Environment.GetEnvironmentVariable("OPENAI_API_KEY"));

    var imageBytes = await File.ReadAllBytesAsync(imagePath);
    var base64 = Convert.ToBase64String(imageBytes);
    var mimeType = imagePath.EndsWith(".png") ? "image/png" : "image/jpeg";

    var messages = new List<ChatMessage>
    {
        new SystemChatMessage(systemPrompt),
        new UserChatMessage(
            ChatMessageContentPart.CreateTextPart(userPrompt),
            ChatMessageContentPart.CreateImagePart(
                new BinaryData(imageBytes), mimeType
            )
        )
    };

    var options = new ChatCompletionOptions
    {
        ResponseFormat = ChatResponseFormat.CreateJsonObjectFormat(),
        MaxOutputTokenCount = 1000,
        Temperature = 0f
    };

    var response = await client.CompleteChatAsync(messages, options);
    return response.Value.Content[0].Text;
}

// ============================================================================
// 2. Self-hosted Modal endpoint (OpenAI-compatible)
// ============================================================================

async Task<string> AnalyzeWithModal(string imagePath, string systemPrompt, string userPrompt)
{
    var credential = new ApiKeyCredential("not-used");
    var options = new OpenAIClientOptions
    {
        Endpoint = new Uri("https://kaustubhkumar05--inference-engine-finetuned-vllmserver-serve.modal.run/v1")
    };

    // Modal proxy auth headers
    var httpClient = new HttpClient();
    httpClient.DefaultRequestHeaders.Add("Modal-Key", Environment.GetEnvironmentVariable("MODAL_TOKEN_ID"));
    httpClient.DefaultRequestHeaders.Add("Modal-Secret", Environment.GetEnvironmentVariable("MODAL_TOKEN_SECRET"));
    options.Transport = new HttpClientPipelineTransport(httpClient);

    var client = new ChatClient("llm", credential, options);

    var imageBytes = await File.ReadAllBytesAsync(imagePath);
    var mimeType = imagePath.EndsWith(".png") ? "image/png" : "image/jpeg";

    var messages = new List<ChatMessage>
    {
        new SystemChatMessage(systemPrompt),
        new UserChatMessage(
            ChatMessageContentPart.CreateTextPart(userPrompt),
            ChatMessageContentPart.CreateImagePart(
                new BinaryData(imageBytes), mimeType
            )
        )
    };

    var chatOptions = new ChatCompletionOptions
    {
        MaxOutputTokenCount = 1000,
        Temperature = 0f
    };

    var response = await client.CompleteChatAsync(messages, chatOptions);
    return response.Value.Content[0].Text;
}

// ============================================================================
// Usage
// ============================================================================

var systemPrompt = "You are a kitchen interior analysis assistant. Respond with JSON only.";
var userPrompt = "Analyze this kitchen image. Identify cabinets, colors, finishes, and handles.";
var imagePath = "kitchen_01.jpg";

// Swap between providers — same interface
var result = await AnalyzeWithOpenAI(imagePath, systemPrompt, userPrompt);
// var result = await AnalyzeWithModal(imagePath, systemPrompt, userPrompt);

Console.WriteLine(result);
