import { StrictMode } from 'react'
import { hydrateRoot } from 'react-dom/client'

import { pageFor } from './pages'
import './styles.css'

const root = document.getElementById('root')
if (!root) throw new Error('no #root')
const { Component } = pageFor(location.pathname)

hydrateRoot(
  root,
  <StrictMode>
    <Component />
  </StrictMode>,
)
