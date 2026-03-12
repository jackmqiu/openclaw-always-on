import { describe, expect, it } from "vitest";
import {
  deriveConversationIdentityFromSessionKey,
  formatMemoriesForPrompt,
  rankMemoriesByConversationContext,
} from "./index.js";

describe("deriveConversationIdentityFromSessionKey", () => {
  it("parses channel direct sessions", () => {
    const identity = deriveConversationIdentityFromSessionKey("agent:main:telegram:direct:alice");
    expect(identity).toMatchObject({
      channel: "telegram",
      chatType: "direct",
      peerId: "alice",
    });
    expect(identity?.entityKey).toContain("telegram");
  });

  it("parses account-scoped direct sessions", () => {
    const identity = deriveConversationIdentityFromSessionKey(
      "agent:main:discord:acct1:direct:user42",
    );
    expect(identity).toMatchObject({
      channel: "discord",
      accountId: "acct1",
      chatType: "direct",
      peerId: "user42",
    });
  });

  it("captures thread suffixes", () => {
    const identity = deriveConversationIdentityFromSessionKey(
      "agent:main:discord:channel:c123:thread:t456",
    );
    expect(identity).toMatchObject({
      channel: "discord",
      chatType: "channel",
      peerId: "c123",
      threadId: "t456",
    });
  });

  it("returns null for malformed keys", () => {
    expect(deriveConversationIdentityFromSessionKey("main")).toBeNull();
  });
});

describe("rankMemoriesByConversationContext", () => {
  it("prioritizes entity matches over higher base score mismatches", () => {
    const results = rankMemoriesByConversationContext(
      [
        {
          id: "1",
          memory: "non-match high score",
          score: 0.95,
          metadata: { conversationEntityKey: "discord:channel:other" },
        },
        {
          id: "2",
          memory: "match lower score",
          score: 0.6,
          metadata: { conversationPeerId: "alice", conversationChannel: "telegram" },
        },
      ],
      "agent:main:telegram:direct:alice",
    );
    expect(results[0]?.id).toBe("2");
  });

  it("keeps original order basis when no session key", () => {
    const input = [
      { id: "1", memory: "a", score: 0.7 },
      { id: "2", memory: "b", score: 0.9 },
    ];
    const results = rankMemoriesByConversationContext(input, undefined);
    expect(results).toEqual(input);
  });
});

describe("formatMemoriesForPrompt", () => {
  it("includes explicit conversation context tags when metadata exists", () => {
    const text = formatMemoriesForPrompt([
      {
        id: "m1",
        memory: "Discussed rollout timeline",
        score: 0.84,
        metadata: {
          conversationEntityKey: "discord:direct:user42",
          conversationChannel: "discord",
          conversationChatType: "direct",
          conversationPeerId: "user42",
          conversationThreadId: "t9",
        },
      },
    ]);
    expect(text).toContain("<relevant-memories>");
    expect(text).toContain("[id:m1]");
    expect(text).toContain("[context:");
    expect(text).toContain("channel=discord");
    expect(text).toContain("chat=direct");
    expect(text).toContain("peer=user42");
    expect(text).toContain("thread=t9");
  });
});
