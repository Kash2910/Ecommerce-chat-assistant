// Import required modules from LangChain ecosystem
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai" // For creating vector embeddings from text using Gemini
import { ChatGoogleGenerativeAI } from "@langchain/google-genai" // Google's Gemini AI model
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages" // Message types for conversations
import {
  ChatPromptTemplate,      // For creating structured prompts with placeholders
  MessagesPlaceholder,     // Placeholder for dynamic message history
} from "@langchain/core/prompts"
import { StateGraph } from "@langchain/langgraph"              // State-based workflow orchestration
import { Annotation } from "@langchain/langgraph"              // Type annotations for state management
import { tool } from "@langchain/core/tools"                   // For creating custom tools/functions
import { ToolNode } from "@langchain/langgraph/prebuilt"       // Pre-built node for executing tools
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb" // For saving conversation state
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb"   // Vector search integration with MongoDB
import { MongoClient } from "mongodb"                          // MongoDB database client
import { z } from "zod"                                        // Schema validation library
import "dotenv/config"                                         // Load environment variables from .env file
import { Db } from "mongodb";

// Utility function to handle API rate limits with exponential backoff
async function retryWithBackoff<T>(
  fn: () => Promise<T>,    // The function to retry (generic type T for return value)
  maxRetries = 3           // Maximum number of retry attempts (default 3)
): Promise<T> {
  // Loop through retry attempts
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await fn()    // Try to execute the function
    } catch (error: any) {
      // Check if it's a rate limit error (HTTP 429) and we have retries left
      if (error.status === 429 && attempt < maxRetries) {
        // Calculate exponential backoff delay: 2^attempt seconds, max 30 seconds
        const delay = Math.min(1000 * Math.pow(2, attempt), 30000)
        console.log(`Rate limit hit. Retrying in ${delay / 1000} seconds...`)
        // Wait for the calculated delay before retrying
        await new Promise(resolve => setTimeout(resolve, delay))
        continue // Go to next iteration (retry)
      }
      throw error // If not rate limit or out of retries, throw the error
    }
  }
  throw new Error("Max retries exceeded") // This should never be reached
}

// Main function that creates and runs the AI agent
export async function callAgent(client: MongoClient, query: string, thread_id: string) {
  try {
    // Database configuration
    const dbName = "inventory_database"        // Name of the MongoDB database
    const db = client.db(dbName)              // Get database instance
    const collection = db.collection("items") // Get the 'items' collection

    // Define the state structure for the agent workflow
    const GraphState = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        // Reducer function: how to combine old and new messages
        reducer: (x, y) => x.concat(y), // Simply concatenate old messages (x) with new messages (y)
      }),
    })

    function cosineSimilarity(a: number[], b: number[]): number {
      const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
      const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
      const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
      return dot / (normA * normB);
    }


    // Create a custom tool for searching furniture inventory
   async function itemLookupTool(query: string, db: Db) {
      const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GOOGLE_API_KEY!,
        model: "text-embedding-004", // Gemini embedding model
      });

      try {
        console.log("Item lookup tool called with query:", query);

        // Step 1: Create embedding for query
        const queryEmbedding = await embeddings.embedQuery(query);

        // Step 2: Fetch all items
        const items = await db.collection("items").find().toArray();
        console.log("Total items fetched:", items.length);

        if (items.length === 0) {
          return "No items found in the database.";
        }

        // Step 3: Calculate similarity
        const scoredItems = items.map((item) => {
          if (!item.embedding) return { ...item, score: -1 };
          return {
            ...item,
            score: cosineSimilarity(queryEmbedding, item.embedding),
          };
        });

        // Step 4: Sort and take top 3
        const topMatches = scoredItems.sort((a, b) => b.score - a.score).slice(0, 3);

        // Step 5: Format response
        return topMatches.map((item) => ({
          id: item.item_id,
          name: item.item_name,
          description: item.item_description,
          score: item.score.toFixed(3),
          price: item.prices?.sale_price ?? item.prices?.full_price,
          categories: item.categories,
        }));
      } catch (err) {
        console.error("Error in item lookup:", err);
        return "An error occurred while looking up items.";
      }
    }


    // Array of all available tools (just one in this case)
    const tools = [itemLookupTool]
    // Create a tool execution node for the workflow
    const toolNode = new ToolNode<typeof GraphState.State>(tools)

    // Initialize the AI model (Google's Gemini)
    const model = new ChatGoogleGenerativeAI({
      model: "gemini-1.5-flash",         //  Use Gemini 1.5 Flash model
      temperature: 0,                    // Deterministic responses (no randomness)
      maxRetries: 0,                     // Disable built-in retries (we handle our own)
      apiKey: process.env.GOOGLE_API_KEY, // Google API key from environment
    }).bindTools(tools)                  // Bind our custom tools to the model

    // Decision function: determines next step in the workflow
    function shouldContinue(state: typeof GraphState.State) {
      const messages = state.messages                               // Get all messages
      const lastMessage = messages[messages.length - 1] as AIMessage // Get the most recent message

      // If the AI wants to use tools, go to tools node; otherwise end
      if (lastMessage.tool_calls?.length) {
        return "tools"  // Route to tool execution
      }
      return "__end__"  // End the workflow
    }

    // Function that calls the AI model with retry logic
    async function callModel(state: typeof GraphState.State) {
      return retryWithBackoff(async () => { // Wrap in retry logic
        // Create a structured prompt template
        const prompt = ChatPromptTemplate.fromMessages([
          [
            "system", // System message defines the AI's role and behavior
            `You are a helpful E-commerce Chatbot Agent for a furniture store. 

IMPORTANT: You have access to an item_lookup tool that searches the furniture inventory database. ALWAYS use this tool when customers ask about furniture items, even if the tool returns errors or empty results.

When using the item_lookup tool:
- If it returns results, provide helpful details about the furniture items
- If it returns an error or no results, acknowledge this and offer to help in other ways
- If the database appears to be empty, let the customer know that inventory might be being updated

Current time: {time}`,
          ],
          new MessagesPlaceholder("messages"), // Placeholder for conversation history
        ])

        // Fill in the prompt template with actual values
        const formattedPrompt = await prompt.formatMessages({
          time: new Date().toISOString(), // Current timestamp
          messages: state.messages,       // All previous messages
        })

        // Call the AI model with the formatted prompt
        const result = await model.invoke(formattedPrompt)
        // Return new state with the AI's response added
        return { messages: [result] }
      })
    }

    // Build the workflow graph
    const workflow = new StateGraph(GraphState)
      .addNode("agent", callModel)                    // Add AI model node
      .addNode("tools", toolNode)                     // Add tool execution node
      .addEdge("__start__", "agent")                  // Start workflow at agent
      .addConditionalEdges("agent", shouldContinue)   // Agent decides: tools or end
      .addEdge("tools", "agent")                      // After tools, go back to agent

    // Initialize conversation state persistence
    const checkpointer = new MongoDBSaver({ client, dbName })
    // Compile the workflow with state saving
    const app = workflow.compile({ checkpointer })

    // Execute the workflow
    const finalState = await app.invoke(
      {
        messages: [new HumanMessage(query)], // Start with user's question
      },
      {
        recursionLimit: 15,                   // Prevent infinite loops
        configurable: { thread_id: thread_id } // Conversation thread identifier
      }
    )

    // Extract the final response from the conversation
    const response = finalState.messages[finalState.messages.length - 1].content
    console.log("Agent response:", response)

    return response // Return the AI's final response

  } catch (error: any) {
    // Handle different types of errors with user-friendly messages
    console.error("Error in callAgent:", error.message)

    if (error.status === 429) { // Rate limit error
      throw new Error("Service temporarily unavailable due to rate limits. Please try again in a minute.")
    } else if (error.status === 401) { // Authentication error
      throw new Error("Authentication failed. Please check your API configuration.")
    } else { // Generic error
      throw new Error(`Agent failed: ${error.message}`)
    }
  }
}
