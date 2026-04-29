/**
 * Message Security and Formatting Utilities
 * Implements highest security standards for chat message rendering
 */

import { marked, Marked, Renderer } from 'marked'
import DOMPurify from 'dompurify'

const renderer = new Renderer()

const markedInstance = new Marked({
  renderer,
  gfm: true,
  breaks: true,
})

/**
 * Escape HTML entities to prevent XSS
 */
export function escapeHtml(text: string): string {
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

/**
 * Sanitize HTML content using DOMPurify
 * Removes all dangerous tags and attributes
 */
export function sanitizeHtml(html: string): string {
  return DOMPurify.sanitize(html, {
    ALLOWED_TAGS: [
      'p', 'br', 'strong', 'b', 'em', 'i', 'u', 'strike', 'del',
      'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
      'ul', 'ol', 'li',
      'blockquote',
      'code', 'pre',
      'a',
      'table', 'thead', 'tbody', 'tr', 'th', 'td',
      'hr'
    ],
    ALLOWED_ATTR: [
      'href', 'title',  // For links
      'class'          // For syntax highlighting
    ],
    ALLOW_DATA_ATTR: false,  // No data-* attributes
    SANITIZE_DOM: true,
    KEEP_CONTENT: true
  })
}

/**
 * Render markdown safely
 * Converts markdown to HTML, then sanitizes
 */
export function renderMarkdown(text: string): string {
  if (!text) return ''
  
  // Parse markdown
  const rawHtml = markedInstance.parse(text) as string
  
  // Sanitize the HTML
  return sanitizeHtml(rawHtml)
}

/**
 * Format message for display
 * User messages: Escape HTML only
 * Assistant messages: Render markdown + code highlighting
 */
export function formatMessage(text: string, role: 'user' | 'assistant'): string {
  if (!text) return ''
  
  if (role === 'user') {
    // For user messages, just escape HTML and preserve line breaks
    return escapeHtml(text).replace(/\n/g, '<br>')
  } else {
    // For assistant messages, render markdown
    return renderMarkdown(text)
  }
}

/**
 * Apply syntax highlighting to code blocks
 * Note: Currently styled via CSS. highlight.js integration would go here.
 */
export function applySyntaxHighlighting(): void {
  // Placeholder for future highlight.js integration
  // Currently code blocks are styled via CSS in the component
}

/**
 * Convert URLs in text to clickable links (for plain text)
 */
export function linkifyText(text: string): string {
  const urlRegex = /(https?:\/\/[^\s<]+[^<.,:;\s])/g
  return text.replace(urlRegex, (url) => {
    return `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(url)}</a>`
  })
}

/**
 * Strip all HTML tags (for maximum safety)
 */
export function stripHtml(text: string): string {
  return DOMPurify.sanitize(text, { ALLOWED_TAGS: [], KEEP_CONTENT: true })
}

/**
 * Validate input for security issues
 * Returns { isValid, error }
 */
export function validateInput(text: string, maxLength: number = 4000): { isValid: boolean; error?: string } {
  if (!text || text.trim().length === 0) {
    return { isValid: false, error: 'Message cannot be empty' }
  }
  
  if (text.length > maxLength) {
    return { isValid: false, error: `Message exceeds maximum length of ${maxLength} characters` }
  }
  
  // Check for dangerous patterns
  const dangerousPatterns = [
    /<script\b/i,
    /<iframe\b/i,
    /<object\b/i,
    /<embed\b/i,
    /javascript\s*:/i,
    /\bon(click|load|error|mouseover|focus|blur|submit|change)\s*=/i,
    /data\s*:\s*text\/html/i
  ]
  
  for (const pattern of dangerousPatterns) {
    if (pattern.test(text)) {
      return { isValid: false, error: 'Message contains prohibited content' }
    }
  }
  
  return { isValid: true }
}

/**
 * Sanitize user input before sending to backend
 */
export function sanitizeInput(text: string): string {
  // Remove null bytes
  text = text.replace(/\x00/g, '')
  
  // Normalize line endings
  text = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n')
  
  // Trim whitespace
  return text.trim()
}