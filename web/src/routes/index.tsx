import { createFileRoute } from '@tanstack/react-router'

import { ChatView } from '@/components/chat/ChatView'
import { Page } from '@/components/layout'
import type { PageMeta } from '@/lib/page'

export const pageMeta: PageMeta = {
  titleKey: 'chat.title',
  descriptionKey: 'chat.description',
}

export const Route = createFileRoute('/')({
  component: ChatPage,
  staticData: { pageMeta },
})

function ChatPage() {
  return (
    <Page>
      <ChatView />
    </Page>
  )
}
